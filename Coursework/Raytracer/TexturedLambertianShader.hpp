#pragma once
#include "Shader.hpp"
#include <algorithm> // Required for std::min/max

/// <summary>
/// Lambertian reflectance shader that samples albedo values from a texture.
/// </summary>
class TexturedLambertianShader : public Shader
{
private:
	const std::vector<uint8_t>* albedoTexture_;
	const int texWidth_, texHeight_;
	bool shadowTest_;

public:
	TexturedLambertianShader(const std::vector<uint8_t>* albedoTexture, int texWidth, int texHeight, bool shadowTest = true)
		: shadowTest_(shadowTest), albedoTexture_(albedoTexture),
		texWidth_(texWidth), texHeight_(texHeight)
	{
	}

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo,
		const Renderable* scene,
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const
	{
		// SAFETY 1: Check if texture pointer is null or empty to prevent instant crash
		if (!albedoTexture_ || albedoTexture_->empty()) {
			return Eigen::Vector3f(0.5f, 0.5f, 0.5f);
		}

		Eigen::Vector3f albedo;
		Eigen::Vector2f tex = hitInfo.texCoords;

		// SAFETY 2: Fix the off-by-one error. Clamp to width-1 and height-1
		int pixX = static_cast<int>(tex.x() * texWidth_);
		int pixY = static_cast<int>((1.f - tex.y()) * texHeight_);

		pixX = std::max(0, std::min(pixX, texWidth_ - 1));
		pixY = std::max(0, std::min(pixY, texHeight_ - 1));

		// SAFETY 3: Final index check before memory access
		size_t baseIdx = static_cast<size_t>(pixX + texWidth_ * pixY) * 4;
		if (baseIdx + 2 >= albedoTexture_->size()) {
			return Eigen::Vector3f(1.0f, 0.0f, 1.0f); // Magenta error color if something is still wrong
		}

		albedo.x() = static_cast<float>((*albedoTexture_)[baseIdx + 0]) / 255.f;
		albedo.y() = static_cast<float>((*albedoTexture_)[baseIdx + 1]) / 255.f;
		albedo.z() = static_cast<float>((*albedoTexture_)[baseIdx + 2]) / 255.f;

		Eigen::Vector3f color = albedo.cwiseProduct(ambientLight);

		for (auto& light : lights) {
			if (shadowTest_) {
				if (!light->visibilityCheck(hitInfo.location, scene))
					continue;
			}
			Eigen::Vector3f lightVec = light->getVecToLight(hitInfo.location);
			float dotProd = std::max(lightVec.dot(hitInfo.normal), 0.f);

			// Using Eigen's standard cwiseProduct for albedo multiplication
			color += dotProd * light->getIntensity(hitInfo.location).cwiseProduct(albedo);
		}

		return color;
	}
};