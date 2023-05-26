#pragma once

#include "util/log.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

#include <cstring>

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

		frame.resize(100, 100);

		// copy data to frame
		for (int i = 0; i < frame.size(); ++i)
			frame[i] = cache[i + 20];

		// rotate frame by transpose and flip
		for (int y = 0; y < frame.shape(1); ++y)
			for (int x = 0; x < y; ++x)
				std::swap(frame(x, y), frame(y, x));

		for (int y = 0; y < frame.shape(1); ++y)
			for (int x = 0; x < frame.shape(0) / 2; ++x)
				std::swap(frame(x, y), frame(frame.shape(0) - x - 1, y));

		return *this;
	}

	sense&
	operator>> (image<rgba>& frame)
	{
		static image<byte> raw;
		(*this) >> raw;

		auto interpolate = [](byte x) {
			if (x == 255)
				return rgba(0, 0, 0, 0);
		};

		for (int y = 0; y < frame.shape(1); ++y)
			for (int x = 0; x < frame.shape(0); ++x)
				frame(x, y) = interpolate(raw(x, y));
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

		_tof_at("AT+BAUD=6");

		cfsetispeed(&tio, B1000000);
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
