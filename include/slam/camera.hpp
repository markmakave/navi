/*

	Copyright (c) 2023 Mokhov Mark

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.

*/

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

#include <cstring>
#include <cerrno>
#include <cstdint>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "slam/device.hpp"
#include "base/color.hpp"
#include "base/image.hpp"

#include "util/timer.hpp"

namespace lumina {
namespace slam {

class camera : public device
{
public:

	camera(const char* filename,
		   unsigned	   width		= 640,
		   unsigned	   height		= 480,
		   int		   buffer_count = 1)
	  : device(filename),
		width(width),
		height(height),
		streaming(false)
	{
		buffers.resize(buffer_count);

		_pass_format();
		_request_buffers();
		_allocate_buffers();
	}

	void
	info() override
	{
		v4l2_capability cap;
		if (ioctl(VIDIOC_QUERYCAP, &cap) != 0) {
			throw std::runtime_error("Camera info request failed");
		}

		std::cout << "Driver:        " << cap.driver << "\n\t"
				  << "Card:          " << cap.card << "\n\t"
				  << "Bus info:      " << cap.bus_info << "\n\t"
				  << "Capabilities:  \n";

		if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
			std::cout << "\t\tv4l2 dev support capture\n";

		if (cap.capabilities & V4L2_CAP_VIDEO_OUTPUT)
			std::cout << "\t\tv4l2 dev support output\n";

		if (cap.capabilities & V4L2_CAP_VIDEO_OVERLAY)
			std::cout << "\t\tv4l2 dev support overlay\n";

		if (cap.capabilities & V4L2_CAP_STREAMING)
			std::cout << "\t\tv4l2 dev support streaming\n";

		if (cap.capabilities & V4L2_CAP_READWRITE)
			std::cout << "\t\tv4l2 dev support read write\n";
	}

	void
	start()
	{
		int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		if (ioctl(VIDIOC_STREAMON, &type) != 0) {
			throw std::runtime_error("Camera stream starting failed");
		}

		v4l2_buffer buf = {};
		buf.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory		= V4L2_MEMORY_MMAP;

		for (buf.index = 0; buf.index < buffers.size(); ++buf.index)
			if (ioctl(VIDIOC_QBUF, &buf) != 0)
				throw std::runtime_error("Camera queuing buffer failed");

		streaming = true;
	}

	void
	stop()
	{
		int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		if (ioctl(VIDIOC_STREAMOFF, &type) != 0) {
			throw std::runtime_error("Camera stream stoping failed");
		}
		streaming = false;
	}

	camera&
	operator>> (image<lumina::gray>& frame)
	{
		if (width != frame.shape()[0] || height != frame.shape()[1]) {
			frame.resize(width, height);
		}

		if (!streaming)
			start();

		v4l2_buffer buf = {};
		buf.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory		= V4L2_MEMORY_MMAP;

		if (ioctl(VIDIOC_DQBUF, &buf) != 0) {
			throw std::runtime_error("Camera buffer dequeuing failed");
		}

		for (int i = 0; i < frame.size(); ++i) {
			frame.data()[i] = buffers[buf.index][i * 2];
		}

		if (ioctl(VIDIOC_QBUF, &buf) != 0) {
			throw std::runtime_error("Camera buffer queuing failed");
		}

		return *this;
	}

	camera&
	operator>> (image<rgb>& frame)
	{
		if (width != frame.shape()[0] || height != frame.shape()[1]) {
			frame.resize(width, height);
		}

		if (!streaming)
			start();

		v4l2_buffer buf = {};
		buf.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory		= V4L2_MEMORY_MMAP;

		if (ioctl(VIDIOC_DQBUF, &buf) != 0) {
			throw std::runtime_error("Camera buffer dequeuing failed");
		}

		for (int i = 0; i < frame.size() / 2; ++i) {
			uint8_t y1 = buffers[buf.index][i * 4 + 0];
			uint8_t u  = buffers[buf.index][i * 4 + 1];
			uint8_t y2 = buffers[buf.index][i * 4 + 2];
			uint8_t v  = buffers[buf.index][i * 4 + 3];

			frame.data()[i * 2 + 0] = rgb(yuv(y1, u, v));
			frame.data()[i * 2 + 1] = rgb(yuv(y2, u, v));
		}

		if (ioctl(VIDIOC_QBUF, &buf) != 0) {
			throw std::runtime_error("Camera buffer queuing failed");
		}

		return *this;
	}

	camera&
	operator>> (array<byte>& frame)
	{
		if (!streaming)
			start();

		v4l2_buffer buf = {};
		buf.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory		= V4L2_MEMORY_MMAP;

		if (ioctl(VIDIOC_DQBUF, &buf) != 0) {
			throw std::runtime_error("Camera buffer dequeuing failed");
		}

		frame.reshape(buf.length);
		std::memcpy(frame.data(), buffers[buf.index], buf.length);

		if (ioctl(VIDIOC_QBUF, &buf) != 0) {
			throw std::runtime_error("Camera buffer queuing failed");
		}

		return *this;
	}

	~camera()
	{
		if (streaming) {
			stop();
		}
		for (auto buffer : buffers) {
			munmap(buffer, width * height * 2);
		}
	}

private:

	void
	_pass_format()
	{
		v4l2_format fmt			= {};
		fmt.type				= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		fmt.fmt.pix.width		= width;
		fmt.fmt.pix.height		= height;
		fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
		fmt.fmt.pix.field		= V4L2_FIELD_NONE;

		if (ioctl(VIDIOC_S_FMT, &fmt) != 0) {
			throw std::runtime_error("Camera format setting failed");
		}
	}

	void
	_request_buffers()
	{
		v4l2_requestbuffers req = {};
		req.count				= buffers.size();
		req.type				= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		req.memory				= V4L2_MEMORY_MMAP;

		if (ioctl(VIDIOC_REQBUFS, &req) != 0) {
			throw std::runtime_error("Camera requesting buffers failed");
		}
	}

	void
	_allocate_buffers()
	{
		v4l2_buffer buf = {};
		buf.type		= V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory		= V4L2_MEMORY_MMAP;

		for (buf.index = 0; buf.index < buffers.size(); ++buf.index) {
			if (ioctl(VIDIOC_QUERYBUF, &buf) != 0) {
				throw std::runtime_error("Camera quering buffer failed");
			}

			buffers[buf.index] =
				static_cast<uint8_t*>(mmap(NULL,
										   buf.length,
										   PROT_READ | PROT_WRITE,
										   MAP_SHARED,
										   fd,
										   buf.m.offset));
			if (buffers[buf.index] == MAP_FAILED) {
				throw std::runtime_error("Camera buffer mapping failed");
			}
		}
	}

private:

	unsigned			  width, height;
	std::vector<uint8_t*> buffers;
	bool				  streaming;
};

} // namespace slam
} // namespace lumina

// bash command to select video exposure in v4l2
// v4l2-ctl -d /dev/video0 -c exposure_auto=1
