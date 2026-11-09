"""Read RTSP video with GStreamer and return owned BGR NumPy frames."""

import logging
from urllib.parse import urlsplit

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import Gst, GstRtsp, GstVideo

log = logging.getLogger("gstreamer")


class GStreamerReader:
    def __init__(
        self,
        rtsp_url: str,
        latency: int = 150,
        timeout_ms: int = 5000,
    ) -> None:
        if latency < 0:
            raise ValueError("latency must not be negative")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0")

        self._validate_rtsp_url(rtsp_url)
        Gst.init(None)

        self.pipeline: Gst.Pipeline | None = None
        self.appsink: Gst.Element | None = None
        self.bus: Gst.Bus | None = None
        self.opened = False
        self.timeout_ms = timeout_ms

        try:
            self._build_pipeline(rtsp_url, latency)
            assert self.pipeline is not None
            result = self.pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Cannot start GStreamer pipeline")
            self.opened = True
        except Exception:
            self.release()
            raise

    @staticmethod
    def _validate_rtsp_url(rtsp_url: str) -> None:
        if not isinstance(rtsp_url, str) or not rtsp_url:
            raise ValueError("RTSP URL must be a non-empty string")
        if any(ord(char) <= 0x20 or ord(char) == 0x7F for char in rtsp_url):
            raise ValueError("RTSP URL must not contain whitespace or control characters")

        try:
            parsed = urlsplit(rtsp_url)
            hostname = parsed.hostname
            parsed.port
        except ValueError as exc:
            raise ValueError("Invalid RTSP URL") from exc

        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not hostname:
            raise ValueError("RTSP URL must use rtsp:// or rtsps:// and include a host")

    def _build_pipeline(self, rtsp_url: str, latency: int) -> None:
        pipeline = Gst.Pipeline.new("rtsp-camera-pipeline")
        if pipeline is None:
            raise RuntimeError("Cannot create GStreamer pipeline")

        source = self._make_element("rtspsrc", "rtsp-source")
        decoder = self._make_element("decodebin", "decoder")
        converter = self._make_element("videoconvert", "video-converter")
        caps_filter = self._make_element("capsfilter", "bgr-format")
        appsink = self._make_element("appsink", "frame-sink")

        source.set_property("location", rtsp_url)
        source.set_property("protocols", GstRtsp.RTSPLowerTrans.TCP)
        source.set_property("latency", latency)
        if source.find_property("tcp-timeout") is not None:
            source.set_property("tcp-timeout", self.timeout_ms * 1000)

        caps_filter.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw,format=BGR"),
        )
        appsink.set_property("emit-signals", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)
        appsink.set_property("wait-on-eos", False)

        for element in (source, decoder, converter, caps_filter, appsink):
            pipeline.add(element)

        if not converter.link(caps_filter) or not caps_filter.link(appsink):
            raise RuntimeError("Cannot link GStreamer video output elements")

        source.connect("pad-added", self._link_video_rtp_pad, decoder)
        decoder.connect("pad-added", self._link_decoded_video_pad, converter)

        self.pipeline = pipeline
        self.appsink = appsink
        self.bus = pipeline.get_bus()

    @staticmethod
    def _make_element(factory_name: str, element_name: str) -> Gst.Element:
        element = Gst.ElementFactory.make(factory_name, element_name)
        if element is None:
            raise RuntimeError(
                f"Required GStreamer element is unavailable: {factory_name}"
            )
        return element

    @staticmethod
    def _link_video_rtp_pad(
        _source: Gst.Element,
        source_pad: Gst.Pad,
        decoder: Gst.Element,
    ) -> None:
        caps = source_pad.get_current_caps() or source_pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return

        structure = caps.get_structure(0)
        if (
            structure.get_name() != "application/x-rtp"
            or structure.get_string("media") != "video"
        ):
            return

        decoder_pad = decoder.get_static_pad("sink")
        if decoder_pad is None or decoder_pad.is_linked():
            return

        result = source_pad.link(decoder_pad)
        if result != Gst.PadLinkReturn.OK:
            log.error("Cannot link RTSP video pad to decoder: %s", result.value_nick)

    @staticmethod
    def _link_decoded_video_pad(
        _decoder: Gst.Element,
        source_pad: Gst.Pad,
        converter: Gst.Element,
    ) -> None:
        caps = source_pad.get_current_caps() or source_pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        if not caps.get_structure(0).get_name().startswith("video/x-raw"):
            return

        converter_pad = converter.get_static_pad("sink")
        if converter_pad is None or converter_pad.is_linked():
            return

        result = source_pad.link(converter_pad)
        if result != Gst.PadLinkReturn.OK:
            log.error("Cannot link decoded video pad to converter: %s", result.value_nick)

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        appsink = self.appsink
        if not self.opened or appsink is None:
            return False, None

        sample = appsink.emit(
            "try-pull-sample",
            self.timeout_ms * Gst.MSECOND,
        )
        if sample is None:
            self._handle_missing_sample()
            return False, None

        caps = sample.get_caps()
        buffer = sample.get_buffer()
        if caps is None or buffer is None:
            log.error("GStreamer returned a sample without caps or buffer")
            return False, None

        video_info = GstVideo.VideoInfo.new_from_caps(caps)
        if video_info is None:
            log.error("Cannot read video information from GStreamer sample caps")
            return False, None

        width = video_info.width
        height = video_info.height
        stride = video_info.stride[0]
        offset = video_info.offset[0]
        if width <= 0 or height <= 0 or stride < width * 3 or offset < 0:
            log.error(
                "Invalid GStreamer BGR frame layout width=%d height=%d stride=%d offset=%d",
                width,
                height,
                stride,
                offset,
            )
            return False, None

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            log.error("Cannot map GStreamer video buffer")
            return False, None

        try:
            required_size = offset + (height - 1) * stride + width * 3
            if required_size > map_info.size:
                log.error(
                    "GStreamer frame buffer is truncated required=%d actual=%d",
                    required_size,
                    map_info.size,
                )
                return False, None

            frame_view = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data,
                offset=offset,
                strides=(stride, 3, 1),
            )
            return True, frame_view.copy()
        except (TypeError, ValueError, BufferError):
            log.exception("Cannot convert GStreamer buffer to BGR frame")
            return False, None
        finally:
            buffer.unmap(map_info)

    def _handle_missing_sample(self) -> None:
        bus = self.bus
        if bus is None:
            self.opened = False
            log.error("GStreamer bus is unavailable")
            return

        message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message is None:
            self.opened = False
            log.error(
                "No GStreamer video sample received within %d ms",
                self.timeout_ms,
            )
            return

        self.opened = False
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            log.error("GStreamer stream error: %s; debug=%s", error, debug)
        else:
            log.warning("GStreamer stream reached end-of-stream")

    def release(self) -> None:
        self.opened = False
        pipeline = self.pipeline
        self.pipeline = None
        self.appsink = None
        self.bus = None
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
