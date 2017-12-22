#pragma once

#define D_FOG_NOISE 1.0f

#define D_STRONG_FOG 0.0f

#define D_UPDATE_TRANS_FIRST 0

#include "../scene/scene.h"

#include "cutils_math.h"

#include "../driver/cuda_helper.h"

#define LPOS make_float3(25.0f,36.0f, -20.0f)
#define LCOL (6000.0f * make_float3( 1.0f, 0.9, 0.5))

__device__ inline float
displacementSimple( float2 p)
{
	return 1.0f;
}

__device__ inline float3
getSceneColor(float3 p, float material)
{
	return make_float3(0.0f);
}

__device__ inline float
getClosestDistance(float3 p)
{
	return 1.0f;
}

__device__ inline float3
evaluateLight(float3 pos)
{
    float3 lightPos = LPOS;
    float3 lightCol = LCOL;
    float3 L = lightPos-pos;
    return lightCol * 1.0f/dot(L,L);
}


__device__ inline float
rand(float n){return fracf(sin(n) * 43758.5453123);}

__device__ inline float
noise(float p)
{
	float fl = floor(p);
	float fc = fracf(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

__device__ inline void
getParticipatingMedia( float& muS,  float& muE,  float3 pos)
{
    float heightFog = 0.0f + D_FOG_NOISE*3.0f;
    heightFog = 0.3*clamp((heightFog-pos.y)*1.0f, 0.0f, 1.0f);
    
    const float fogFactor = 1.0f + D_STRONG_FOG * 5.0f;
    
	float d = 27.7f;
    const float3 sphereRadius = make_float3(8.f, 5.5f, 5.5f);
	float disp = sin(pos.x + d)*cos(pos.y + d)*sin(pos.z + d)*1.2f*noise(pos.x);
    //float sphereFog = clamp(length(make_float2(length(make_float2(off.x, off.z) - sphereRadius), off.y)), 0.0f,1.0f);
    float sphereFog = clamp(min(min(sphereRadius.x, sphereRadius.y), sphereRadius.z) * (1.f - length((pos - make_float3(20.0f,19.0f,-17.0f)) / sphereRadius)) + disp, 0.f, 1.f);
    
    float constantFog = 0.02f;

    muS = constantFog + heightFog * fogFactor + sphereFog;
   
    const float muA = 0.0f;
    muE = max(0.000000001, muA + muS);
}

__device__ inline float
isotropic(float k, float costh)
{
    return 1.0f / (4.0f * 3.14f);
}

__device__ inline float
schlick(float k, float costh)
{
    return (1.0 - k * k) / (4.0 * M_PI * pow(1.0 - k * costh, 2.0));
}

__device__ inline float
rayleigh(float k, float costh)
{
	return 3.f / 16.f * (1.f + costh * costh);
}

__device__ inline float
phaseFunction(float t, float costh)
{
	return isotropic(t, costh);
	//return schlick(t, costh);
}

__device__ inline float
volumetricShadow( float3 from,  float3 to)
{
    const float numStep = 8.0f;
    float shadow = 1.0f;
    float muS = 0.0f;
    float muE = 0.0f;
    float dd = length(to-from) / numStep;
    for(float s = 0.5f; s < (numStep-0.1); s += 1.0f)
    {
        float3 pos = from + (to - from) * (s / numStep);
        getParticipatingMedia(muS, muE, pos);
        shadow *= exp(-muE * dd);
    }
    return shadow;
}

__device__ inline void
volume_raymarch(scene::Ray& r, float3& albedo, float4& scatTrans)
{
	const int numIter = 100;

	float muS = 0.0f;
	float muE = 0.0f;
	float3 lightPos = LPOS;

	float transmittance = 1.0f;
	float3 scatteredLight = make_float3(0.0f, 0.0f, 0.0f);
	float material = 0.0f;

	float d = 1.0f;
	float3 p = make_float3(0.0f, 0.0f, 0.0f);
	float dd = 0.0f;
	float3 n = make_float3(0.0f);
	for (int i = 0; i<numIter; ++i)
	{
		float3 p = r.origin + d * r.dir;

		getParticipatingMedia(muS, muE, p);

		float cd = getClosestDistance(p);
		float g = 0.7f;
		float k = 1.55f * g - 0.55f * g * g * g;
		float3 p2 = r.origin + (d + cd) * r.dir;
		//float costh = dot(p, p2) / (length(p) * length(p2));
		//float costh = dot(make_float3(1.0f), p) / length(p) * length(make_float3(1.0f));
		float costh = p.x / length(p);
		// Frostbite optimization
		float3 S = evaluateLight(p) * muS * phaseFunction(k, costh) * volumetricShadow(p, lightPos);
		float3 Sint = (S - S * exp(-muE * dd)) / muE;
		scatteredLight = scatteredLight + transmittance * Sint;

		transmittance *= exp(-muE * dd);

		dd = cd;
		if (dd < 0.2f)
			break;
		d += dd;
	}

	scatTrans = make_float4(scatteredLight, transmittance);
}
