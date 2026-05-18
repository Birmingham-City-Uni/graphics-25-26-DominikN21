#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream> 
#include <string>
#include <Eigen/Dense>
#include <lodepng.h>

#include "BVHNode.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "PointLight.hpp"
#include "TexturedLambertianShader.hpp"
#include "Model.hpp"
#include "MirrorShader.hpp"

int main(int argc, char* argv[]) {
    const int pixWidth = 1920;
    const int pixHeight = 1080;

    // --- CAMERA ---
    Eigen::Vector3f camPos(-167.0f, 8.0f, -185.0f);
    Eigen::Vector3f target(-150.0f, 8.0f, -100.0f);
    Eigen::Vector3f forward = (target - camPos).normalized();
    Camera cam(camPos, forward, Eigen::Vector3f(0, 1, 0), pixWidth, pixHeight, 80.0f);

    std::vector<uint8_t> outImage(pixHeight * pixWidth * 4, 0);

    // --- TEXTURE LOADING ---
    auto texPattern = new std::vector<uint8_t>();
    auto texFloor = new std::vector<uint8_t>();
    auto texWall = new std::vector<uint8_t>();
    auto texWater = new std::vector<uint8_t>();
    unsigned wP, hP, wF, hF, wW, hW, wWa, hWa;

    lodepng::decode(*texPattern, wP, hP, "models/Pattern.png");
    lodepng::decode(*texFloor, wF, hF, "models/Floor.png");
    lodepng::decode(*texWall, wW, hW, "models/Wall.png");
    lodepng::decode(*texWater, wWa, hWa, "models/Water.png");

    // --- INITIALIZE SHADERS ---
    TexturedLambertianShader patternShader(texPattern, wP, hP);
    TexturedLambertianShader floorShader(texFloor, wF, hF);
    TexturedLambertianShader wallShader(texWall, wW, hW);
    TexturedLambertianShader waterShader(texWater, wWa, hWa);
    MirrorShader mirrorShader;

    Scene scene;
    auto addModel = [&](std::string path, Shader* s) {
        try {
            Model* m = new Model(path.c_str());
            scene.renderables.push_back(std::make_shared<BVHNode>(*m, s, 4, Eigen::Matrix4f::Identity()));
            std::cout << "Successfully Loaded: " << path << std::endl;
        }
        catch (...) { std::cout << "Load failed for: " << path << std::endl; }
        };

    // --- LOAD ALL MODELS ---
    addModel("models/LeftWall3.obj", &patternShader);
    addModel("models/Floor.obj", &mirrorShader);
    addModel("models/MidWall.obj", &wallShader);
    addModel("models/RightWall.obj", &patternShader);
    addModel("models/Roof.obj", &floorShader);
    addModel("models/SmallWall.obj", &patternShader);
    addModel("models/TinyWall.obj", &wallShader);
    addModel("models/Pool.obj", &waterShader);
    addModel("models/Pillar1.obj", &mirrorShader);
    addModel("models/Pillar2.obj", &mirrorShader);
    addModel("models/Pillar3.obj", &mirrorShader);
    addModel("models/Pillar4.obj", &mirrorShader);

    std::vector<std::unique_ptr<Light>> lights;
    lights.push_back(std::make_unique<PointLight>(Eigen::Vector3f(-160.f, 40.f, -140.f), Eigen::Vector3f(15000.f, 15000.f, 15000.f)));

    std::cout << "Rendering Scene with all models loaded..." << std::endl;

#pragma omp parallel for
    for (int y = 0; y < pixHeight; ++y) {
        for (int x = 0; x < pixWidth; ++x) {
            Ray ray = cam.getRay(x, (pixHeight - y) - 1);
            HitInfo hit;

            bool hitSomething = scene.intersect(ray, 0.1f, 5000.0f, hit, 0xFFFFFFFF);
            if (!hitSomething) {
                Ray inv = ray; inv.direction = -ray.direction;
                hitSomething = scene.intersect(inv, 0.1f, 5000.0f, hit, 0xFFFFFFFF);
            }

            if (hitSomething && hit.shader != nullptr) {
                Eigen::Vector3f color = hit.shader->getColor(hit, &scene, lights, Eigen::Vector3f(0.1f, 0.1f, 0.1f), 0, 2);

                int idx = (y * pixWidth + x) * 4;
                outImage[idx + 0] = (uint8_t)(std::min(1.0f, color.x()) * 255);
                outImage[idx + 1] = (uint8_t)(std::min(1.0f, color.y()) * 255);
                outImage[idx + 2] = (uint8_t)(std::min(1.0f, color.z()) * 255);
                outImage[idx + 3] = 255;
            }
        }
    }

    lodepng::encode("raytrace_output.png", outImage, pixWidth, pixHeight);
    std::cout << "Render complete." << std::endl;

    return 0;
}