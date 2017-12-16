#pragma once

#define D_FOG_NOISE 1.0

#define D_STRONG_FOG 0.0

#define D_UPDATE_TRANS_FIRST 0

#include "../scene/scene.h"

#include "cutils_math.h"

#include "../driver/cuda_helper.h"

#define LPOS make_float3( 20.0+15.0, 11.0+0.0,-20.0) + make_float3(-10.0,25.0, 0.0)
#define LCOL (600.0*make_float3( 1.0, 0.9, 0.5))  * 10.0

__device__ inline float
displacementSimple( float2 p)
{
	return 1.0f;
}

__device__ inline float3
getSceneColor(float3 p, float material)
{
	return make_float3(0.0);
}

__device__ inline float
getClosestDistance(float3 p,  float& material)
{
    float minD = 1.0;
	material = 0.0;
    
	return minD;
}

__device__ inline float3
evaluateLight(float3 pos)
{
    float3 lightPos = LPOS;
    float3 lightCol = LCOL;
    float3 L = lightPos-pos;
    return lightCol * 1.0/dot(L,L);
}

__device__ inline float3
evaluateLight( float3 pos, float3 normal)
{
    float3 lightPos = LPOS;
    float3 L = lightPos-pos;
    float distanceToL = length(L);
    float3 Lnorm = L/distanceToL;
    return max(0.0,dot(normal,Lnorm)) * evaluateLight(pos);
}

__device__ inline void
getParticipatingMedia( float& muS,  float& muE,  float3 pos)
{
    float disp = clamp(displacementSimple(make_float2(pos.x, pos.z)*0.005f + 0.01),0.0,1.0);
    float heightFog = 0.0 + D_FOG_NOISE*3.0*disp;
    heightFog = 0.3*clamp((heightFog-pos.y)*1.0f, 0.0, 1.0);
    
    const float fogFactor = 1.0 + D_STRONG_FOG * 5.0;
    
    const float sphereRadius = 5.0;
    float sphereFog = clamp((sphereRadius-length(pos-make_float3(20.0,19.0,-17.0)))/sphereRadius, 0.0,1.0)*disp;
    
    float constantFog = 0.02;

    muS = constantFog + heightFog*fogFactor + sphereFog;
   
    const float muA = 0.0;
    muE = max(0.000000001, muA + muS); // to avoid division by zero extinction
}

__device__ inline float
phaseFunction()
{
    return 1.0/(4.0*3.14);
}

__device__ inline float
volumetricShadow( float3 from,  float3 to)
{
    const float numStep = 8.0; // quality control. Bump to avoid shadow alisaing
    float shadow = 1.0;
    float muS = 0.0;
    float muE = 0.0;
    float dd = length(to-from) / numStep;
    for(float s=0.5; s<(numStep-0.1); s+=1.0)// start at 0.5 to sample at center of integral part
    {
        float3 pos = from + (to-from)*(s/(numStep));
        getParticipatingMedia(muS, muE, pos);
        shadow *= exp(-muE * dd);
    }
    return shadow;
}

__device__ inline void
volume_raymarch(scene::Ray& r, float3& albedo, float4& scatTrans)
{
	const int numIter = 60;

	float muS = 0.0f;
	float muE = 0.0f;
	float3 lightPos = LPOS;

	float transmittance = 1.0f;
	float3 scatteredLight = make_float3(0.0, 0.0, 0.0);
	float material = 0.0;

	float d = 1.0f;
	float3 p = make_float3(0.0, 0.0, 0.0);
	float dd = 0.0f;
	float3 n = make_float3(0.0);
	for (int i = 0; i<numIter; ++i)
	{
		float3 p = r.origin + d * r.dir;

		getParticipatingMedia(muS, muE, p);

		if (true)
		{
			float3 S = evaluateLight(p) * muS * phaseFunction() * volumetricShadow(p, lightPos);// incoming light
			float3 Sint = (S - S * exp(-muE * dd)) / muE; // integrate along the current step segment
			scatteredLight = scatteredLight + transmittance * Sint; // accumulate and also take into account the transmittance from previous steps

													// Evaluate transmittance to view independentely
			transmittance *= exp(-muE * dd);
		}

		dd = getClosestDistance(p, material);
		if (dd < 0.2f)
			break; // give back a lot of performance without too much visual loss
		d += dd;
	}

	scatTrans = make_float4(scatteredLight, transmittance);
}
