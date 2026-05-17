// This define is necessary to get the M_PI constant.
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <lodepng.h>

// If you get "nothing works" errors, make sure the very first line 
// inside Mesh.hpp, Light.hpp, and LinAlg.hpp is: #pragma once
#include "LinAlg.hpp"
#include "Image.hpp"
#include "Light.hpp"
#include "Mesh.hpp"

// Define the struct BEFORE the functions use it
struct Triangle {
	std::array<Eigen::Vector3f, 3> screen;
	std::array<Eigen::Vector3f, 3> verts;
	std::array<Eigen::Vector3f, 3> norms;
	std::array<Eigen::Vector2f, 3> texs;
};

// ========= Subtask 1: Make a Projection Matrix ========
Eigen::Matrix4f projectionMatrix(int height, int width, float horzFov = 70.f * M_PI / 180.f, float zFar = 1000.f, float zNear = 0.00001f)
{
	float aspect = (float)width / (float)height;
	float vertFov = 2.0f * atan(tan(horzFov / 2.0f) / aspect);
	float f = 1.0f / tan(vertFov / 2.0f);

	Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
	projection << 1.f / tanf(horzFov * 0.5f), 0, 0, 0,
		0, 1.f / tanf(vertFov * .5f), 0, 0,
		0, 0, zFar / (zFar - zNear), -(zFar * zNear) / (zFar - zNear),
		0, 0, 1.f, 0;
	return projection;
}

void findScreenBoundingBox(const Triangle& t, int width, int height, int& minX, int& minY, int& maxX, int& maxY)
{
	minX = (int)std::min({ t.screen[0].x(), t.screen[1].x(), t.screen[2].x() });
	minY = (int)std::min({ t.screen[0].y(), t.screen[1].y(), t.screen[2].y() });
	maxX = (int)std::max({ t.screen[0].x(), t.screen[1].x(), t.screen[2].x() });
	maxY = (int)std::max({ t.screen[0].y(), t.screen[1].y(), t.screen[2].y() });

	minX = std::max(0, std::min(minX, width - 1));
	maxX = std::max(0, std::min(maxX, width - 1));
	minY = std::max(0, std::min(minY, height - 1));
	maxY = std::max(0, std::min(maxY, height - 1));
}

// drawTriangle MUST be above drawMesh
void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	std::vector<float>& zBuffer, const Triangle& t,
	const std::vector<std::unique_ptr<Light>>& lights,
	const std::vector<uint8_t>& albedoTexture, int texWidth, int texHeight)
{
	int minX, minY, maxX, maxY;
	findScreenBoundingBox(t, width, height, minX, minY, maxX, maxY);

	Eigen::Vector2f edge1 = v2(t.screen[2] - t.screen[0]);
	Eigen::Vector2f edge2 = v2(t.screen[1] - t.screen[0]);
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) return;

	for (int x = minX; x <= maxX; ++x) {
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p((float)x, (float)y);

			float a0 = 0.5f * fabsf(vec2Cross(v2(t.screen[1]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a1 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a2 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[1]), p - v2(t.screen[1])));

			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			if ((b0 + b1 + b2) > 1.0001f) continue;

			// ========== Subtask 4: Z Buffering ==========
			float depth = b0 * t.screen[0].z() + b1 * t.screen[1].z() + b2 * t.screen[2].z();
			int depthIdx = y * width + x;

			if (depth > zBuffer[depthIdx]) continue;
			zBuffer[depthIdx] = depth;

			Eigen::Vector3f worldP = t.verts[0] * b0 + t.verts[1] * b1 + t.verts[2] * b2;
			Eigen::Vector3f normP = (t.norms[0] * b0 + t.norms[1] * b1 + t.norms[2] * b2).normalized();

			// ========== Subtask 5: Texture Mapping ===========
			Eigen::Vector2f texP = t.texs[0] * b0 + t.texs[1] * b1 + t.texs[2] * b2;
			int texC = std::max(0, std::min((int)(texP.x() * texWidth), texWidth - 1));
			int texR = std::max(0, std::min((int)((1.0f - texP.y()) * texHeight), texHeight - 1));

			Color texColor = getPixel(albedoTexture, texC, texR, texWidth, texHeight);
			Eigen::Vector3f albedo(powf(texColor.r / 255.f, 2.2f), powf(texColor.g / 255.f, 2.2f), powf(texColor.b / 255.f, 2.2f));

			Eigen::Vector3f color = Eigen::Vector3f::Zero();

			for (auto& light : lights) {
				Eigen::Vector3f lightIntensity = light->getIntensityAt(worldP);
				if (light->getType() != Light::Type::AMBIENT) {
					float dotProd = std::max(normP.dot(-light->getDirection(worldP)), 0.0f);
					lightIntensity *= dotProd;
				}
				color += coeffWiseMultiply(lightIntensity, albedo);
			}

			Color c;
			c.r = std::min(powf(color.x(), 1 / 2.2f), 1.0f) * 255;
			c.g = std::min(powf(color.y(), 1 / 2.2f), 1.0f) * 255;
			c.b = std::min(powf(color.z(), 1 / 2.2f), 1.0f) * 255;
			c.a = 255;
			setPixel(image, x, y, width, height, c);
		}
	}
}

// drawMesh MUST be above main
void drawMesh(std::vector<unsigned char>& image, std::vector<float>& zBuffer, const Mesh& mesh,
	const std::vector<uint8_t>& albedoTexture, int texWidth, int texHeight,
	const Eigen::Matrix4f& modelToWorld, const Eigen::Matrix4f& worldToClip,
	const std::vector<std::unique_ptr<Light>>& lights, int width, int height)
{
	for (int i = 0; i < (int)mesh.vFaces.size(); ++i) {
		Triangle t;
		t.verts[0] = (modelToWorld * vec3ToVec4(mesh.verts[mesh.vFaces[i][0]])).block<3, 1>(0, 0);
		t.verts[1] = (modelToWorld * vec3ToVec4(mesh.verts[mesh.vFaces[i][1]])).block<3, 1>(0, 0);
		t.verts[2] = (modelToWorld * vec3ToVec4(mesh.verts[mesh.vFaces[i][2]])).block<3, 1>(0, 0);

		// ======= Subtask 2: The Transformation Chain ======
		Eigen::Vector4f vClip[3];
		bool skipTriangle = false;
		for (int j = 0; j < 3; ++j) {
			vClip[j] = worldToClip * vec3ToVec4(t.verts[j]);
			if (vClip[j].w() != 0) vClip[j] /= vClip[j].w();

			// If even one point is outside the clip box, we flag it (though a true rasterizer would clip the triangle)
			if (outsideClipBox(vClip[j])) skipTriangle = true;

			t.screen[j] = Eigen::Vector3f(
				(vClip[j].x() + 1.f) * 0.5f * width,
				(1.f - (vClip[j].y() + 1.f) * 0.5f) * height,
				vClip[j].z()
			);
		}
		if (skipTriangle) continue;

		t.norms[0] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * mesh.norms[mesh.nFaces[i][0]]).normalized();
		t.norms[1] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * mesh.norms[mesh.nFaces[i][1]]).normalized();
		t.norms[2] = (modelToWorld.block<3, 3>(0, 0).inverse().transpose() * mesh.norms[mesh.nFaces[i][2]]).normalized();

		t.texs[0] = mesh.texs[mesh.tFaces[i][0]];
		t.texs[1] = mesh.texs[mesh.tFaces[i][1]];
		t.texs[2] = mesh.texs[mesh.tFaces[i][2]];

		drawTriangle(image, width, height, zBuffer, t, lights, albedoTexture, texWidth, texHeight);
	}
}

int main()
{
	std::string outputFilename = "output.png";
	const int width = 1920, height = 1080;
	const int nChannels = 4;

	std::vector<uint8_t> imageBuffer(height * width * nChannels);
	std::vector<float> zBuffer(height * width);

	Color black{ 0,0,0,255 };
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			setPixel(imageBuffer, c, r, width, height, black);
			zBuffer[r * width + c] = 1e9f;
		}
	}

	// ========== Subtask 3: Camera Matrices ========
	Eigen::Matrix4f projection = projectionMatrix(height, width);
	Eigen::Matrix4f cameraToWorld = translationMatrix(Eigen::Vector3f(-181.f, 6.0f, -190.f)) * rotateXMatrix(0.4f);

	// Set up worldToCamera and worldToClip
	Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();
	Eigen::Matrix4f worldToClip = projection * worldToCamera;

	std::string bunnyFilename = "../models/LeftWall2.obj";
	std::string textureFilename = "../models/Wall.png";

	std::vector<std::unique_ptr<Light>> lights;
	lights.emplace_back(new AmbientLight(Eigen::Vector3f(0.5f, 0.5f, 0.5f)));
	lights.emplace_back(new DirectionalLight(Eigen::Vector3f(1.0f, 1.0f, 1.0f), Eigen::Vector3f(1.f, 0.f, 0.0f)));

	Mesh bunnyMesh = loadMeshFile(bunnyFilename);
	if (bunnyMesh.verts.empty()) {
		std::cout << "ERROR: Cannot load mesh. Check if " << bunnyFilename << " exists." << std::endl;
		return -1;
	}

	std::vector<uint8_t> bunnyTexture;
	unsigned int bunnyTexWidth, bunnyTexHeight;
	lodepng::decode(bunnyTexture, bunnyTexWidth, bunnyTexHeight, textureFilename);

	// Draw the bunnies
	for (float x : {-1.0f, 1.0f}) {
		for (float z : {-3.0f, -5.0f, -7.0f}) {
			Eigen::Matrix4f wallTransform = Eigen::Matrix4f::Identity();
			drawMesh(imageBuffer, zBuffer, bunnyMesh, bunnyTexture, bunnyTexWidth, bunnyTexHeight,
				wallTransform, worldToClip, lights, width, height);
		}
	}

	// Debug lights
	drawPointLights(imageBuffer, width, height, lights);

	// Save Output
	int errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) {
		std::cout << "lodepng error encoding image: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	saveZBufferImage("zBuffer.png", zBuffer, width, height);
	std::cout << "Render successful! Saved to output.png and zBuffer.png" << std::endl;

	return 0;
}