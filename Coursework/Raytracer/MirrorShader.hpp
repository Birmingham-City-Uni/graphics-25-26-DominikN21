#pragma once
#include "Shader.hpp"
#include "GeomUtil.hpp"

/// <summary>
/// Shader modelling perfect mirror reflectance.
/// </summary>
class MirrorShader : public Shader
{
public:

	virtual Eigen::Vector3f getColor(const HitInfo& hitInfo, 
		const Renderable* scene, 
		const std::vector<std::unique_ptr<Light>>& lights,
		const Eigen::Vector3f& ambientLight,
		int currBounceCount,
		const int maxBounces) const
	{
		if (currBounceCount >= maxBounces) return Eigen::Vector3f::Zero();

		Ray reflectionRay;
		reflectionRay.direction = reflect(hitInfo.inDirection, hitInfo.normal);
		reflectionRay.origin = hitInfo.location + 0.1f * hitInfo.normal;

		Eigen::Vector3f color = Eigen::Vector3f::Zero();

		HitInfo reflectionHit;
		if (scene->intersect(reflectionRay, 1e-6f, 1e4f, reflectionHit, VISIBLE_BITMASK)) {
			color = reflectionHit.shader->getColor(
				reflectionHit, scene, 
				lights, ambientLight, 
				currBounceCount+1, maxBounces);
		}

		return color;
	}
};
