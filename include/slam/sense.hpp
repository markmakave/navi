#pragma once

#include "util/log.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

#include <cstring>
#include <cassert>

#if __cplusplus >= 201703L

namespace lm::slam {

class sense
{
public:

	sense(const char* stereo_dev, const char* tof_dev)
	{
		// _stereo_fd = open(stereo_dev, O_RDWR);
		_stereo_fd = 0;
		_tof_fd	   = open(tof_dev, O_RDWR);

		if (_stereo_fd < 0 || _tof_fd < 0) {
			log::error("Device open failed:", strerror(errno));
			return;
		}

		_configure_stereo();
		_configure_tof();
	}

	~sense()
	{}

	sense&
	operator>> (image<byte>& frame)
	{
		int				   packet_length = 10022;
		static array<byte> cache(packet_length * 2);
		int				   border = 0;

		frame.reshape(100, 100);

		// fill cache with full packet
		while (true) {
			int n = read(_tof_fd, cache.data() + border, packet_length);
			assert(n >= 0);
			if (n == 0)
				continue;

			border += n;

			if (border >= packet_length)
				break;
		}

		// search for header and shift data to the beginning
		for (int i = 0; i < packet_length - 1; ++i) {
			if (cache[i] == 0x00 && cache[i + 1] == 0xFF) {
				if (*(short*)(&cache[i + 2]) != 10016) {
					log::error("mismatch");
					continue;
				}
				// shift data
				for (int j = i; j < border; ++j)
					cache[j - i] = cache[j];

				border -= i;

				break;
			}
		}

		// read the rest of the packet
		while (border < packet_length) {
			int n = read(_tof_fd, cache.data() + border, packet_length - border);
			assert(n >= 0);
			border += n;
		}

		log::info("frame ID:", *(unsigned short*)(cache.data() + 16));

		// copy data to frame
		for (int i = 0; i < frame.size(); ++i)
			frame[i] = cache[i + 20];

		// rotate frame
		for (int y = 0; y < frame.shape(1) / 2; ++y)
			for (int x = 0; x < frame.shape(0) / 2; ++x) {
				byte backup = frame(x, y);

				frame(x, y) = frame(y, frame.shape(0) - 1 - x);
				frame(y, frame.shape(0) - 1 - x) =
					frame(frame.shape(0) - 1 - x, frame.shape(1) - 1 - y);

				frame(frame.shape(0) - 1 - x, frame.shape(1) - 1 - y) =
					frame(frame.shape(1) - 1 - y, x);
				frame(frame.shape(1) - 1 - y, x) = backup;
			}

		return *this;
	}

	sense&
	operator>> (image<rgba>& frame)
	{
		static image<byte> raw;
		(*this) >> raw;

		auto interpolate = [](byte x) -> rgba {
			if (x == 255)
				return rgba(0, 0, 0, 0);

			float factor = x / 255.0f * 2.0f;

			if (factor < 1.f) {
				return rgba(255 * (1.f - factor), 255 * factor, 0, 255);
			} else {
				factor -= 1.f;
				return rgba(0, 255 * (1.f - factor), 255 * factor, 255);
			}
		};

		frame.reshape(raw.shape());

		for (int i = 0; i < raw.size(); ++i)
			frame[i] = interpolate(raw[i]);

		return *this;
	}

private:

	void
	_configure_stereo()
	{}

	void
	_configure_tof()
	{
		struct termios tio;
		tcgetattr(_tof_fd, &tio);
		cfmakeraw(&tio);
		tio.c_iflag &= ~(IXON | IXOFF);
		tio.c_cc[VTIME] = 0;
		tio.c_cc[VMIN]	= 0;

		cfsetspeed(&tio, B115200);
		int err = tcsetattr(_tof_fd, TCSAFLUSH, &tio);
		assert(err == 0);

		_tof_at("AT+BAUD=8");

		cfsetispeed(&tio, B3000000);
		err = tcsetattr(_tof_fd, TCSAFLUSH, &tio);
		assert(err == 0);

		_tof_at("AT+ISP=0");
		_tof_at("AT+DISP=4");
		_tof_at("AT+ISP=1");
	}

	void
	_tof_at(const char* cmd)
	{
		write(_tof_fd, cmd, std::strlen(cmd));
		write(_tof_fd, "\r\n", 2);
	}

private:

	int _stereo_fd;
	int _tof_fd;
};

} // namespace lm::slam

#endif
