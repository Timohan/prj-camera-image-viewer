//
// Created by timo on 14.4.2021.
//

#ifndef MATH3D_H
#define MATH3D_H

#include <math.h>

#define PI_VALUE 3.141593f
#define START_AREA_PLUS_VALUE 10.0f
typedef struct
{
    float m[16];
} Matrix4x4;

typedef struct
{
    float x;
    float y;
    float z;
} VECTOR3;


typedef struct
{
    float x;
    float y;
} VECTOR2;

typedef struct
{
    int x;
    int y;
} CGPoint;

class Math3D {
public:
    static float getDistance(float x0, float y0, float x1, float y1);
    static float getDistance3D(float x0 , float y0 , float z0 , float x1 , float y1 , float z1);

    static float getHeight(float x0 , float y0 , float z0 , float x1 , float y1 , float z1 , float x , float z);
    static void calculateLineRatio(float *fRatioX , float *fRatioY , float *fRatioC , const VECTOR2 vPointFrom , const VECTOR2  vPointTo);
    static bool calculateCrossPoint(const VECTOR2 vPointFrom1 , const VECTOR2  vPointTo1 , const VECTOR2 vPointFrom2 , const VECTOR2  vPointTo2, VECTOR2 * rvVector);
    static void setMatrixValue(Matrix4x4 *m , int y , int x , float fValue);
    static float getMatrixValue(const Matrix4x4 *m , int y , int x);

    static void MatrixIdentity(Matrix4x4 *m);

    static void SetOrtho( Matrix4x4 *m , float left , float right , float bottom , float top , float near , float far);
    static void SetRotX( Matrix4x4 *m , float angle);
    static void SetRotZ( Matrix4x4 *m , float angle);
    static void SetRotY( Matrix4x4 *m , float angle);
    static void Mul( Matrix4x4 *m , const Matrix4x4 *a , const Matrix4x4 *b);

    static float calcAngleY( const VECTOR3 end);
    static float calcAngleY( const VECTOR3 start , const VECTOR3 end);
    static float calcAngleXZ( const VECTOR3 end);
    static float calcAngleXZ(const VECTOR3 start , const VECTOR3 end);
    static void setAngleXZ( float *rvX , float *rvZ , float fAngle);
    static float getYFromAngleY(float x, float z, const float fAngleY);

    static float Vec3Dot( const VECTOR3 pV1 , const VECTOR3 pV2);
    static VECTOR3 decreaceVector(VECTOR3 value0 , VECTOR3 value1);
    static VECTOR3 Vec3Cross(const VECTOR3 pV1 , const VECTOR3 pV2);
    static VECTOR3 Vec3Normalize(const VECTOR3 value);

    static float determinate(const Matrix4x4 *m);
    static VECTOR3 vectorMatrix(const VECTOR3 *vector , const Matrix4x4 *m);
    static void inversion( const Matrix4x4 *inputMatrix , Matrix4x4 *dst);
    static void MatrixLookAtLH(Matrix4x4 *outMatrix , const VECTOR3 pEye , const VECTOR3 pAt , const VECTOR3 pUp);

    static VECTOR3 calculatePointFromLineByOrigDistance(const VECTOR3 vLinePoint , const VECTOR3 vLineDir , float vLineStartDistance , const VECTOR3 vOrig, const float vOrigWantDistance);

    static VECTOR3 moveDistance( const VECTOR3 pOrig , const VECTOR3 vDir , float fDistance);
    static VECTOR3 moveDistance2( const VECTOR3 vStart , const VECTOR3 vEnd , float fDistance);

    static void pick(const CGPoint ptCursor, const int iWidth, const int iHeight , const Matrix4x4 *proj , const Matrix4x4 *view , VECTOR3 *vDir , VECTOR3 *vOrig);
    static bool doesThisHitToTriangle(const VECTOR3 orig , const VECTOR3 dir , const VECTOR3 v0 , const VECTOR3 v1 , const VECTOR3 v2 , float *t , float * u , float *v , VECTOR3 *outTg , float *distance);
    static void MatrixPerspectiveFovLH( Matrix4x4 *pOut , float fovy , float Aspect , float zn , float zf);
    static void MatrixTranslation( Matrix4x4 *outMatrix , float x , float y , float z);
    static void MatrixOrthoLH( Matrix4x4 *pOut , float w , float h , float zn , float zf);

    static void SetRotateX( Matrix4x4 *m , float angle);
    static void SetRotateY( Matrix4x4 *m , float angle);
    static void SetRotateZ(Matrix4x4 *m , float angle);
    static void SetRotateZYX( Matrix4x4 *m , float angleX , float angleY , float angleZ);
    static void SetRotateXZY( Matrix4x4 *m , float angleX , float angleY , float angleZ);
    static void SetRotateYXZ( Matrix4x4 *m , float angleX , float angleY , float angleZ);
    static VECTOR3 calculateAroundBallAngle( const VECTOR3 vector , float angle);

    static float calculateAngle(float x , float y);
    static float calculateAngleDifference(float x1 , float y1 , float x2 , float y2);
    static float calculateClosestAngleDifference( float fAngle0 , float fAngle1);
    static bool isAngleBetween( float fFromAngle , float fToAngle , float fAngleBetween);

    static float fixAngle( float fAngle);
    static void rotateByAngle( float *fOutX , float *fOutY , float x , float y , float fAngle);

    static VECTOR2 getPositionFromOrigDir(VECTOR3 vOrig, VECTOR3 vDir, float fHeight);

    static bool isAngleForward(float fAngle , float fToAngle);
    static bool isAngleForward (float x , float y , float fToAngle);

    static float getGravityTanget(float fGravity);


// set 2D position to 3D ball (Ball Size 1
    static VECTOR3 set2Dto3DBall(float fPixelX, float fPixelY, float fMaxSizeX, float fMaxSizeY);
};


#endif //FIND_FROM_MAP_3D_CMATH3D_H
