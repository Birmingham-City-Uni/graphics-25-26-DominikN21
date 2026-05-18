
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <lodepng.h>
#include "Image.hpp"
#include "LinAlg.hpp"
#include "Light.hpp"
#include "Mesh.hpp"

struct Triangle {
	std::array<Eigen::Vector3f, 3> screen;
	std::array<Eigen::Vector3f, 3> verts;
	std::array<Eigen::Vector3f, 3> norms;
	std::array<Eigen::Vector2f, 3> texs;
};

Eigen::Matrix4f projectionMatrix(int height, int width, float horzFov = 70.f * M_PI / 180.f, float zFar = 10000.0f, float zNear = 0.000001f)
{
	float aspect = (float)width / (float)height;
	float vertFov = 2.0f * atan(tan(horzFov / 2.0f) / aspect);

	float f = 1.0f / tan(vertFov / 2.0f);

	Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
	projection(0, 0) = f / aspect;
	projection(1, 1) = f;
	projection(2, 2) = (zFar + zNear) / (zNear - zFar);
	projection(2, 3) = (2 * zFar * zNear) / (zNear - zFar);
	projection(3, 2) = -1.0f;

	return projection;
}

void findScreenBoundingBox(const Triangle& t, int width, int height, int& minX, int& minY, int& maxX, int& maxY)
{
	minX = std::min(std::min(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	minY = std::min(std::min(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());
	maxX = std::max(std::max(t.screen[0].x(), t.screen[1].x()), t.screen[2].x());
	maxY = std::max(std::max(t.screen[0].y(), t.screen[1].y()), t.screen[2].y());

	minX = std::min(std::max(minX, 0), width - 1);
	maxX = std::min(std::max(maxX, 0), width - 1);
	minY = std::min(std::max(minY, 0), height - 1);
	maxY = std::min(std::max(maxY, 0), height - 1);
}
// caluclates triangles and pixels, checks depth and gets texture colour and lights
void drawTriangle(std::vector<uint8_t>& image, int width, int height,
	std::vector<float>& zBuffer,
	const Triangle& t,
	const std::vector<std::unique_ptr<Light>>& lights,
	const std::vector<uint8_t>& albedoTexture, int texWidth, int texHeight)
{
	int minX, minY, maxX, maxY;
	findScreenBoundingBox(t, width, height, minX, minY, maxX, maxY);

	Eigen::Vector2f edge1 = v2(t.screen[2] - t.screen[0]);
	Eigen::Vector2f edge2 = v2(t.screen[1] - t.screen[0]);
	float triangleArea = 0.5f * vec2Cross(edge2, edge1);
	if (triangleArea < 0) triangleArea = -triangleArea;

	for (int x = minX; x <= maxX; ++x)
		for (int y = minY; y <= maxY; ++y) {
			Eigen::Vector2f p(x, y);

			float a0 = 0.5f * fabsf(vec2Cross(v2(t.screen[1]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a1 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[2]), p - v2(t.screen[2])));
			float a2 = 0.5f * fabsf(vec2Cross(v2(t.screen[0]) - v2(t.screen[1]), p - v2(t.screen[1])));

			float b0 = a0 / triangleArea;
			float b1 = a1 / triangleArea;
			float b2 = a2 / triangleArea;

			float sum = b0 + b1 + b2;
			if (sum > 1.0001) continue;

			Eigen::Vector3f worldP = t.verts[0] * b0 + t.verts[1] * b1 + t.verts[2] * b2;
			float depth = t.screen[0].z() * b0 + t.screen[1].z() * b1 + t.screen[2].z() * b2;

			int depthIdx = y * width + x;
			if (depth > zBuffer[depthIdx]) continue;
			zBuffer[depthIdx] = depth;

			Eigen::Vector3f normP = (t.norms[0] * b0 + t.norms[1] * b1 + t.norms[2] * b2).normalized();
			Eigen::Vector2f texP = t.texs[0] * b0 + t.texs[1] * b1 + t.texs[2] * b2;

			
			float u = texP.x() - floor(texP.x());
			float v = texP.y() - floor(texP.y());

			int texC = (int)(u * (texWidth - 1));
			int texR = (int)((1.0f - v) * (texHeight - 1));

			texC = std::max(0, std::min(texC, texWidth - 1));
			texR = std::max(0, std::min(texR, texHeight - 1));

			Color texColor = getPixel(albedoTexture, texC, texR, texWidth, texHeight);

			Eigen::Vector3f albedo;
			albedo << texColor.r / 255.0f, texColor.g / 255.0f, texColor.b / 255.0f;
			albedo = albedo.array().pow(2.2f);

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

void drawMesh(std::vector<unsigned char>& image,
	std::vector<float>& zBuffer,
	const Mesh& mesh,
	const std::vector<uint8_t>& albedoTexture, int texWidth, int texHeight,
	const Eigen::Matrix4f& modelToWorld,
	const Eigen::Matrix4f& worldToClip,
	const std::vector<std::unique_ptr<Light>>& lights,
	int width, int height)
{
	//camera position
	Eigen::Vector3f cameraPos(-164.0f, 7.0f, -180.0f);

	for (int i = 0; i < mesh.vFaces.size(); ++i) {

		Eigen::Vector3f
			v0_local = mesh.verts[mesh.vFaces[i][0]],
			v1_local = mesh.verts[mesh.vFaces[i][1]],
			v2_local = mesh.verts[mesh.vFaces[i][2]];

		Eigen::Vector3f v0_world = (modelToWorld * vec3ToVec4(v0_local)).head<3>();
		Eigen::Vector3f v1_world = (modelToWorld * vec3ToVec4(v1_local)).head<3>();
		Eigen::Vector3f v2_world = (modelToWorld * vec3ToVec4(v2_local)).head<3>();

		//backface culling 
		Eigen::Vector3f edge1 = v1_world - v0_world;
		Eigen::Vector3f edge2 = v2_world - v0_world;
		Eigen::Vector3f faceNormal = edge1.cross(edge2).normalized();
		Eigen::Vector3f viewDir = (cameraPos - v0_world).normalized();

		if (faceNormal.dot(viewDir) <= 0.0f) {
			continue;
		}

		Triangle t;
		t.verts[0] = v0_world;
		t.verts[1] = v1_world;
		t.verts[2] = v2_world;

		Eigen::Vector4f vClip0 = worldToClip * vec3ToVec4(t.verts[0]);
		Eigen::Vector4f vClip1 = worldToClip * vec3ToVec4(t.verts[1]);
		Eigen::Vector4f vClip2 = worldToClip * vec3ToVec4(t.verts[2]);

		vClip0 /= vClip0.w();
		vClip1 /= vClip1.w();
		vClip2 /= vClip2.w();

		if (outsideClipBox(vClip0) && outsideClipBox(vClip1) && outsideClipBox(vClip2)) {
			continue;
		}

		auto toScreen = [&](Eigen::Vector4f v) {
			float x = (v.x() + 1.0f) * 0.5f * width;
			float y = (v.y() + 1.0f) * 0.5f * height;
			return Eigen::Vector3f(x, y, v.z());
			};

		t.screen[0] = toScreen(vClip0);
		t.screen[1] = toScreen(vClip1);
		t.screen[2] = toScreen(vClip2);

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

	Color bg{ 30,30,30,255 };
	for (int r = 0; r < height; ++r)
		for (int c = 0; c < width; ++c) {
			setPixel(imageBuffer, c, r, width, height, bg);
			zBuffer[r * width + c] = 1e9f;
		}

	Eigen::Matrix4f projection = projectionMatrix(height, width, 85.f * M_PI / 180.f, 1000.f);

	// scene is at Z = -208ish
    Eigen::Matrix4f camPos = translationMatrix(Eigen::Vector3f(-164.0f, 7.0f, -180.f));
	float cameraYaw = 3.5f;
	Eigen::Matrix4f camRot = rotateYMatrix(cameraYaw);
	Eigen::Matrix4f cameraToWorld = camPos * camRot;
	Eigen::Matrix4f worldToCamera = cameraToWorld.inverse();

	Eigen::Matrix4f worldToClip = projection * worldToCamera;
	std::vector<std::unique_ptr<Light>> lights;
	lights.emplace_back(new AmbientLight(Eigen::Vector3f(0.6f, 0.6f, 0.6f)));
	lights.emplace_back(new DirectionalLight(Eigen::Vector3f(0.6f, 0.6f, 0.6f), Eigen::Vector3f(1.f, -0.5f, -1.0f)));

	Mesh CharacterMesh1 = loadMeshFile("../models/LeftWall3.obj");
	std::vector<uint8_t> CharacterTexture1;
	unsigned int CharacterTexWidth1, CharacterTexHeight1;
	lodepng::decode(CharacterTexture1, CharacterTexWidth1, CharacterTexHeight1, "../models/Pattern.png");

	Mesh CharacterMesh2 = loadMeshFile("../models/Floor.obj");
	std::vector<uint8_t> CharacterTexture2;
	unsigned int CharacterTexWidth2, CharacterTexHeight2;
	lodepng::decode(CharacterTexture2, CharacterTexWidth2, CharacterTexHeight2, "../models/Floor.png");

	Mesh CharacterMesh3 = loadMeshFile("../models/MidWall.obj");
	std::vector<uint8_t> CharacterTexture3;
	unsigned int CharacterTexWidth3, CharacterTexHeight3;
	lodepng::decode(CharacterTexture3, CharacterTexWidth3, CharacterTexHeight3, "../models/Wall.png");

	Mesh CharacterMesh4 = loadMeshFile("../models/Roof.obj");
	std::vector<uint8_t> CharacterTexture4;
	unsigned int CharacterTexWidth4, CharacterTexHeight4;
	lodepng::decode(CharacterTexture4, CharacterTexWidth4, CharacterTexHeight4, "../models/Floor.png");

	Mesh CharacterMesh5 = loadMeshFile("../models/SmallWall.obj");
	std::vector<uint8_t> CharacterTexture5;
	unsigned int CharacterTexWidth5, CharacterTexHeight5;
	lodepng::decode(CharacterTexture5, CharacterTexWidth5, CharacterTexHeight5, "../models/Pattern.png");

	Mesh CharacterMesh6 = loadMeshFile("../models/TinyWall.obj");
	std::vector<uint8_t> CharacterTexture6;
	unsigned int CharacterTexWidth6, CharacterTexHeight6;
	lodepng::decode(CharacterTexture6, CharacterTexWidth6, CharacterTexHeight6, "../models/Wall.png");

	Mesh CharacterMesh7 = loadMeshFile("../models/Pillar1.obj");
	std::vector<uint8_t> CharacterTexture7;
	unsigned int CharacterTexWidth7, CharacterTexHeight7;
	lodepng::decode(CharacterTexture7, CharacterTexWidth7, CharacterTexHeight7, "../models/Floor.png");

	Mesh CharacterMesh8 = loadMeshFile("../models/Pillar2.obj");
	std::vector<uint8_t> CharacterTexture8;
	unsigned int CharacterTexWidth8, CharacterTexHeight8;
	lodepng::decode(CharacterTexture8, CharacterTexWidth8, CharacterTexHeight8, "../models/Floor.png");

	Mesh CharacterMesh9 = loadMeshFile("../models/Pillar3.obj");
	std::vector<uint8_t> CharacterTexture9;
	unsigned int CharacterTexWidth9, CharacterTexHeight9;
	lodepng::decode(CharacterTexture9, CharacterTexWidth9, CharacterTexHeight9, "../models/Floor.png");

	Mesh CharacterMesh10 = loadMeshFile("../models/Pillar4.obj");
	std::vector<uint8_t> CharacterTexture10;
	unsigned int CharacterTexWidth10, CharacterTexHeight10;
	lodepng::decode(CharacterTexture10, CharacterTexWidth10, CharacterTexHeight10, "../models/Floor.png");

	Mesh CharacterMesh11 = loadMeshFile("../models/Pool.obj");
	std::vector<uint8_t> CharacterTexture11;
	unsigned int CharacterTexWidth11, CharacterTexHeight11;
	lodepng::decode(CharacterTexture11, CharacterTexWidth11, CharacterTexHeight11, "../models/Water.png");


	Eigen::Matrix4f CharacterTransform = Eigen::Matrix4f::Identity();

	drawMesh(imageBuffer, zBuffer, CharacterMesh1, CharacterTexture1, CharacterTexWidth1, CharacterTexHeight1, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh2, CharacterTexture2, CharacterTexWidth2, CharacterTexHeight2, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh3, CharacterTexture3, CharacterTexWidth3, CharacterTexHeight3, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh4, CharacterTexture4, CharacterTexWidth4, CharacterTexHeight4, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh5, CharacterTexture5, CharacterTexWidth5, CharacterTexHeight5, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh6, CharacterTexture6, CharacterTexWidth6, CharacterTexHeight6, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh7, CharacterTexture7, CharacterTexWidth7, CharacterTexHeight7, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh8, CharacterTexture8, CharacterTexWidth8, CharacterTexHeight8, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh9, CharacterTexture9, CharacterTexWidth9, CharacterTexHeight9, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh10, CharacterTexture10, CharacterTexWidth10, CharacterTexHeight10, CharacterTransform, worldToClip, lights, width, height);
	drawMesh(imageBuffer, zBuffer, CharacterMesh11, CharacterTexture11, CharacterTexWidth11, CharacterTexHeight11, CharacterTransform, worldToClip, lights, width, height);




	int errorCode = lodepng::encode(outputFilename, imageBuffer, width, height);
	if (errorCode) {
		std::cout << "lodepng error: " << lodepng_error_text(errorCode) << std::endl;
		return errorCode;
	}

	saveZBufferImage("zBuffer.png", zBuffer, width, height);
	std::cout << "Render saved to output.png!" << std::endl;

	return 0;
}