#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <array>

#include <cstring>
#include <cstdint>
#include <cerrno>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

struct point
{
    float x, y, z;
    uint8_t r, g, b, a;

    point()
    {
        static std::random_device rd;
        static std::mt19937 mt(rd());
        // static std::uniform_int_distribution<uint8_t> color(0, 255);
        static std::normal_distribution<float> cord(0.0, 0.5);

        x = cord(mt);
        y = cord(mt);
        z = cord(mt);

        // colors are sin waves
        r = (uint8_t)(127.0 * sin(x * 2.0 * M_PI) + 128.0);
        g = (uint8_t)(127.0 * sin(y * 2.0 * M_PI) + 128.0);
        b = (uint8_t)(127.0 * sin(z * 2.0 * M_PI) + 128.0);

        a = 255;
    }

    point(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        : x(x), y(y), z(z), r(r), g(g), b(b), a(a)
    {}

    friend
    std::ostream& operator<<(std::ostream& os, const point& p)
    {
        os << "{ " << p.x << ", " << p.y << ", " << p.z << ", "
           << (int)p.r << ", " << (int)p.g << ", " << (int)p.b << ", "
           << (int)p.a << " }";
        return os;
    }
};

int main()
{
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "socket() failed" << std::endl;
        return 1;
    }

    sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, "/tmp/hertz_points.sock");

    if (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "connect() failed" << std::endl;
        // print errno string
        std::cerr << strerror(errno) << std::endl;
        return 1;
    }

    // pack for 1000 points
    std::array<point, 1000> points;

    unsigned i = 0;
    while (true) {

        // pack points into 1000 at a time
        for (auto& p : points) {
            p = point();
        }

        // send points
        if (send(sock, points.data(), sizeof(point) * points.size(), 0) < 0) {
            std::cerr << "send() failed" << std::endl;
            return 1;
        }
        
        std::cout << "sent " << (i += 1000) << '\r' << std::flush;
    }

    close(sock);

    return 0;
}
