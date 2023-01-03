#pragma once

#include <slam/camera.hpp>
#include <slam/pointcloud.hpp>
#include <slam/brief.hpp>
#include <slam/fast.hpp>

#include <base/image.hpp>
#include <base/array.hpp>
#include <base/color.hpp>
#include <base/socket.hpp>

#ifdef DEBUG

void*
operator new(std::size_t size)
{
    std::cout << "Allocating " << size << " bytes" << std::endl;
    return malloc(size);
}

void
operator delete(void* ptr) noexcept
{
    std::cout << "Freeing " << ptr << std::endl;
    free(ptr);
}

#endif

namespace lm {
namespace slam {

template <typename T>
void
concat(const lm::image<T>& l, const lm::image<T>& r, lm::image<T>& out)
{
    if (l.height() != r.height())
        throw std::runtime_error("image height mismatch");

    out.resize(l.width() + r.width(), l.height());

    #pragma omp parallel for
    for (unsigned y = 0; y < l.height(); ++y)
    {
        auto out_row = out.row(y);
        auto l_row = l.row(y);
        auto r_row = r.row(y);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                for (unsigned x = 0; x < l.width(); ++x)
                    out_row[x] = l_row[x];
            }

            #pragma omp section
            {
                for (unsigned x = 0; x < r.width(); ++x)
                    out_row[x + l.width()] = r_row[x];
            }
        }
    }
}

class slam
{

public:

    slam()
    :   _l_camera("/dev/video0", 160, 120),
        _r_camera("/dev/video1", 160, 120),
        _brief(6),
        _fast(20),
        _point_socket("/tmp/hertz_point.sock"),
        _video_socket("/tmp/hertz_video.sock"),
        _command_socket("/tmp/hertz_command.sock")
    {}

    void
    run()
    {
        _l_camera.start();
        _r_camera.start();

        while (true)
        {
            _capture();

            _detect();

            concat(_l_rgb, _r_rgb, _concat);

            _match();

            _encoded = _concat.encode<image<rgb>::format::jpeg>();

            _video_socket.send(_encoded);
        }
    }

private:

    void
    _capture()
    {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                _l_camera >> _l_rgb;
                _l_gray = _l_rgb;
            }

            #pragma omp section
            {
                _r_camera >> _r_rgb;
                _r_gray = _r_rgb;
            }
        }
    }

    void
    _detect()
    {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                _fast.detect(_l_gray, _l_features);
            }

            #pragma omp section
            {
                _fast.detect(_r_gray, _r_features);
            }
        }
    }

    void
    _match()
    {
        _depth.resize(_l_rgb.width(), _l_rgb.height());
        _depth.fill(0);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                for (auto& feature : _l_features)
                {
                    _brief.compute(_l_rgb, feature);
                }
            }

            #pragma omp section
            {
                for (auto& feature : _r_features)
                {
                    _brief.compute(_r_rgb, feature);
                }
            }
        }

        #pragma omp parallel for
        for (auto& l_feature : _l_features)
        {
            auto best_match = _r_features.end();
            unsigned best_distance = 256;

            for (auto r_feature = _r_features.begin(); r_feature != _r_features.end(); ++r_feature)
            {
                // check if row deviation is too large
                if (abs((int)l_feature.y - (int)r_feature->y) > 5)
                    continue;
                    
                auto distance = _brief.distance(l_feature.descriptor, r_feature->descriptor);
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_match = r_feature;
                }
            }

            if (best_match != _r_features.end() && best_distance < 50)
            {
                auto l_x = (int)l_feature.x;
                auto l_y = (int)l_feature.y;
                auto r_x = (int)best_match->x;
                auto r_y = (int)best_match->y;

                _concat.line(l_x, l_y, r_x + _l_rgb.width(), r_y, lm::rgb::random());
            }
        }
    }

    void
    _triangulate()
    {}

    void
    _update()
    {}

private:

    lm::slam::camera _l_camera, _r_camera;
    lm::image<lm::rgb> _l_rgb, _r_rgb;
    lm::image<lm::gray> _l_gray, _r_gray;
    lm::array<lm::slam::feature> _l_features, _r_features;

    lm::slam::brief<256> _brief;
    lm::slam::fast _fast;
    lm::slam::pointcloud _pointcloud;

    // test
    lm::image<lm::rgb> _concat;
    lm::array<uint8_t> _encoded;
    lm::image<lm::gray> _depth;

    lm::socket _point_socket, _video_socket, _command_socket;

};
    
} // namespace slam
} // namespace lm

// set camera exposure using v4l2-ctl
// v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=100
