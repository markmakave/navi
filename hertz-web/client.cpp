#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <string>
#include <vector>

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
        static std::uniform_int_distribution<uint8_t> color(0, 255);
        static std::normal_distribution<float> cord(0.0, 0.5);

        x = cord(mt);
        y = cord(mt);
        z = cord(mt);
        r = color(mt);
        g = color(mt);
        b = color(mt);
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
    strcpy(addr.sun_path, "/tmp/pointcloud.sock");

    if (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "connect() failed" << std::endl;
        // print errno string
        std::cerr << strerror(errno) << std::endl;
        return 1;
    }

    unsigned i = 0;
    while (true) {
        point p;
        if (write(sock, &p, sizeof(point)) < 0) {
            std::cerr << "write() failed" << std::endl;
            return 1;
        }
        std::cout << "sent point " << ++i << ": " << p << std::endl;

        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    close(sock);

    return 0;
}
