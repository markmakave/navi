#include <vpi/LensDistortionModels.h>
#include <vpi/algo/Remap.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include "base/image.hpp"
#include "util/timer.hpp"

#include <cstdlib>

#define ITERATIONS 5

#define VPI_CHECK(x) { VPIStatus s = x; if (s != VPI_SUCCESS) printf("error %s on line %d\n", vpiStatusGetName(s), __LINE__); }

int main()
{
    lm::image<lm::rgb> image("../../../dataset/photo.png");

    // create input image
    VPIImageData data;
    data.format = VPI_IMAGE_FORMAT_RGB8;
    data.numPlanes = 1;
    data.planes[0] = VPIImagePlane{
        VPI_PIXEL_TYPE_3U8,
        image.width(),
        image.height(),
        image.width() * sizeof(lm::rgb),
        image.data()
    };
    VPIImage input;
    VPI_CHECK(vpiImageCreateHostMemWrapper(&data, 0, &input));

    // get input image dimentions and type
    int32_t width = image.width(), height = image.height();

    // create output image with the same dimentions and type
    VPIImage output;
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_RGB8, 0, &output));

    VPIWarpMap map;
    std::memset(&map, 0, sizeof(map));
    map.grid.numHorizRegions  = 1;
    map.grid.numVertRegions   = 1;
    map.grid.regionWidth[0]   = width;
    map.grid.regionHeight[0]  = height;
    map.grid.horizInterval[0] = 1;
    map.grid.vertInterval[0]  = 1;
    VPI_CHECK(vpiWarpMapAllocData(&map));

    VPIFisheyeLensDistortionModel fisheye;
    std::memset(&fisheye, 0, sizeof(fisheye));
    fisheye.mapping = VPI_FISHEYE_EQUIDISTANT;
    fisheye.k1      = 1;
    fisheye.k2      = 0;
    fisheye.k3      = 0;
    fisheye.k4      = 0;

    float sensorWidth = 22.2; /* APS-C sensor */
    float focalLength = 4.0;
    float f = focalLength*width/sensorWidth;
    const VPICameraIntrinsic K =
    {
        { f, 0, width/2.0 },
        { 0, f, height/2.0 }
    };
    const VPICameraExtrinsic X =
    {
        { 1, 0, 0, 0 },
        { 0, 1, 0, 0 },
        { 0, 0, 1, 0 }
    };

    VPI_CHECK(vpiWarpMapGenerateFromFisheyeLensDistortionModel(K, X, K, &fisheye, &map));

    VPIPayload warp;
    VPI_CHECK(vpiCreateRemap(VPI_BACKEND_CUDA, &map, &warp));

    VPIStream stream;
    VPI_CHECK(vpiStreamCreate(0, &stream));

    {
        lm::timer _("compute", ITERATIONS);

        for (int i = 0; i < ITERATIONS; ++i)
        {
            VPI_CHECK(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, warp, input, output, VPI_INTERP_CATMULL_ROM, VPI_BORDER_ZERO, 0));
            VPI_CHECK(vpiStreamSync(stream));
        }
    }

    VPI_CHECK(vpiImageLock(output, VPI_LOCK_READ, &data));

    std::cout << data.numPlanes << ' ' << data.planes[0].pitchBytes << std::endl;

    auto pitch = data.planes[0].pitchBytes;

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            image[y][x] = *( (lm::rgb*)((uint8_t*)data.planes[0].data + y * pitch) + x);
        }

    image.write("out.qoi");
}