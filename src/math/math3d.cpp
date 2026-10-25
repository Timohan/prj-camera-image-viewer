//
// Created by timo on 14.4.2021.
//

#include "math3d.h"

float Math3D::getHeight(float  x0 , float  y0 , float  z0 , float  x1 , float  y1 , float  z1 , float  x , float  z)
{
if (y0 == y1)
{
return y0;
}

if (x0 == x && z0 == z)
{
return y0;
}

if (x1 == x && z1 == z)
{
return y1;
}

float fDiffZ0;
float fDiffZ1;
float fDiffY = y1-y0;

if (x0 == x1)
{
fDiffZ0 = z1 - z0;
fDiffZ1 = z - z0;
}
else {
fDiffZ0 = x1 - x0;
fDiffZ1 = x - x0;
}

return fDiffY*fDiffZ1/fDiffZ0 + y0;
}


void Math3D::calculateLineRatio(float *fRatioX , float *fRatioY , float *fRatioC , const VECTOR2 vPointFrom , const VECTOR2 vPointTo)
{
VECTOR2 vPointFrom2;
VECTOR2 vPointTo2;

if (vPointFrom.x < vPointTo.x)
{
vPointFrom2 = vPointFrom;
vPointTo2 = vPointTo;
}
else
{
vPointFrom2 = vPointTo;
vPointTo2 = vPointFrom;
}

VECTOR2 vPointRatio; //= vPointTo2 - vPointFrom2;

vPointRatio.x = vPointTo2.x - vPointFrom2.x;
vPointRatio.y = vPointTo2.y - vPointFrom2.y;


*fRatioY = 1;
*fRatioX = vPointRatio.y/vPointRatio.x;
if (vPointRatio.x == 0 || vPointRatio.y == 0)
{
*fRatioC = 0;
}
else
{
*fRatioC = (vPointFrom.y - vPointRatio.y*vPointFrom.x/vPointRatio.x);
}


}

bool Math3D::calculateCrossPoint(const VECTOR2 vPointFrom1 , const VECTOR2 vPointTo1 , const VECTOR2 vPointFrom2 , const VECTOR2 vPointTo2 , VECTOR2 *rvVector)
{
VECTOR2 rv;
float fRatioX1, fRatioY1, fRatioC1;
float fRatioX2, fRatioY2, fRatioC2;
float fD=0, fDx, fDy;
bool bOkX;
bool bOkY;
calculateLineRatio(&fRatioX1,&fRatioY1,&fRatioC1,vPointFrom1,vPointTo1);
calculateLineRatio(&fRatioX2,&fRatioY2,&fRatioC2,vPointFrom2,vPointTo2);

fDx = fRatioC1*fRatioY2 - fRatioC2*fRatioY1;
fDy = fRatioX1*fRatioC2 - fRatioX2*fRatioC1;
fD = fRatioX1*fRatioY2 - fRatioX2*fRatioY1;


if (vPointFrom1.x != vPointTo1.x &&
        vPointFrom1.y != vPointTo1.y &&
        vPointFrom2.x != vPointTo2.x &&
        vPointFrom2.y != vPointTo2.y)
{

rv.x = -fDx/fD;
rv.y = fDy/fD;
}
else
{
bOkX = false;
if (vPointFrom1.x == vPointTo1.x)
{
rv.x = vPointTo1.x;
bOkX = true;
}
else
{
if (vPointFrom2.x == vPointTo2.x)
{
rv.x = vPointTo2.x;
bOkX = true;
}
else
{
rv.x = fDx;
}
}
bOkY = false;
if (vPointFrom1.y == vPointTo1.y)
{
rv.y = vPointTo1.y;
bOkY = true;
}
else
{
if (vPointFrom2.y == vPointTo2.y)
{
rv.y = vPointTo2.y;
bOkY = true;
}
else
{
rv.y = fDy;
}
}

if (bOkX != bOkY)
{
if (bOkY == false)
{
if (vPointFrom1.x == vPointTo1.x)
{
fDx = vPointFrom2.x - vPointTo2.x;
fDy = vPointFrom2.y - vPointTo2.y;
fD = vPointFrom2.x - vPointTo1.x;
rv.y = - fD*fDy/fDx + vPointFrom2.y;
}
else
{
fDx = vPointFrom1.x - vPointTo1.x;
fDy = vPointFrom1.y - vPointTo1.y;
fD = vPointFrom1.x - vPointTo2.x;
rv.y = - fD*fDy/fDx + vPointFrom1.y;
}
}
else
{
if (vPointFrom1.y == vPointTo1.y)
{
fDx = vPointFrom2.y - vPointTo2.y;
fDy = vPointFrom2.x - vPointTo2.x;
fD = vPointFrom2.y - vPointTo1.y;
rv.x = - fD*fDy/fDx + vPointFrom2.x;
}
else
{
fDx = vPointFrom1.y - vPointTo1.y;
fDy = vPointFrom1.x - vPointTo1.x;
fD = vPointFrom1.y - vPointTo2.y;
rv.x = - fD*fDy/fDx + vPointFrom1.x;
}
}
}
}

rvVector->x = rv.x;
rvVector->y = rv.y;

return true;
}


float Math3D::getMatrixValue(const Matrix4x4 *m , int y , int x)
{
    return m->m[y*4+x];
}

void Math3D::setMatrixValue(Matrix4x4 *m , int y , int x , float  fValue)
{
    m->m[y*4+x] = fValue;
}

void Math3D::MatrixIdentity(Matrix4x4 *m)
{
    int x, y;

    for (x=0;x<4;x++)
    {
        for (y=0;y<4;y++)
        {
            if (x==y)
            {
                setMatrixValue(m, y, x, 1.0f);
            }
            else {
                setMatrixValue(m, y, x, 0.0f);
            }
        }
    }

}



void Math3D::SetRotY(Matrix4x4 *m, float  angle)
{
const float c = cosf(angle);
const float s = sinf(angle);

m->m[0] = c;
m->m[1] = 0;
m->m[2] = -s;
m->m[3] = 0;

m->m[4] = 0;
m->m[5] = 1;
m->m[6] = 0;
m->m[7] = 0;

m->m[8] = s;
m->m[9] = 0;
m->m[10] = c;
m->m[11] = 0;

m->m[12] = 0;
m->m[13] = 0;
m->m[14] = 0;
m->m[15] = 1;

}

void Math3D::SetRotZ(Matrix4x4 *m , float  angle)
{
const float c = cosf(angle);
const float s = sinf(angle);

m->m[0] = c;
m->m[1] = -s;
m->m[2] = 0;
m->m[3] = 0;

m->m[4] = s;
m->m[5] = c;
m->m[6] = 0;
m->m[7] = 0;

m->m[8] = 0;
m->m[9] = 0;
m->m[10] = 1;
m->m[11] = 0;

m->m[12] = 0;
m->m[13] = 0;
m->m[14] = 0;
m->m[15] = 1;
}

void Math3D::SetRotX( Matrix4x4 *m , float  angle)
{
const float c = cosf(angle);
const float s = sinf(angle);

m->m[0] = 1;
m->m[1] = 0;
m->m[2] = 0;
m->m[3] = 0;

m->m[4] = 0;
m->m[5] = c;
m->m[6] = -s;
m->m[7] = 0;

m->m[8] = 0;
m->m[9] = 0;
m->m[10] = s;
m->m[11] = c;

m->m[12] = 0;
m->m[13] = 0;
m->m[14] = 0;
m->m[15] = 1;
}

void Math3D::Mul( Matrix4x4 *m , const Matrix4x4 *a , const Matrix4x4 *b)
{
m->m[0] = a->m[0]*b->m[0] + a->m[1]*b->m[4] + a->m[2]*b->m[8] + a->m[3]*b->m[12];
m->m[1] = a->m[0]*b->m[1] + a->m[1]*b->m[5] + a->m[2]*b->m[9] + a->m[3]*b->m[13];
m->m[2] = a->m[0]*b->m[2] + a->m[1]*b->m[6] + a->m[2]*b->m[10] + a->m[3]*b->m[14];
m->m[3] = a->m[0]*b->m[3] + a->m[1]*b->m[7] + a->m[2]*b->m[11] + a->m[3]*b->m[15];

m->m[4] = a->m[4]*b->m[0] + a->m[5]*b->m[4] + a->m[6]*b->m[8] + a->m[7]*b->m[12];
m->m[5] = a->m[4]*b->m[1] + a->m[5]*b->m[5] + a->m[6]*b->m[9] + a->m[7]*b->m[13];
m->m[6] = a->m[4]*b->m[2] + a->m[5]*b->m[6] + a->m[6]*b->m[10] + a->m[7]*b->m[14];
m->m[7] = a->m[4]*b->m[3] + a->m[5]*b->m[7] + a->m[6]*b->m[11] + a->m[7]*b->m[15];

m->m[8] = a->m[8]*b->m[0] + a->m[9]*b->m[4] + a->m[10]*b->m[8] + a->m[11]*b->m[12];
m->m[9] = a->m[8]*b->m[1] + a->m[9]*b->m[5] + a->m[10]*b->m[9] + a->m[11]*b->m[13];
m->m[10] = a->m[8]*b->m[2] + a->m[9]*b->m[6] + a->m[10]*b->m[10] + a->m[11]*b->m[14];
m->m[11] = a->m[8]*b->m[3] + a->m[9]*b->m[7] + a->m[10]*b->m[11] + a->m[11]*b->m[15];

m->m[12] = a->m[12]*b->m[0] + a->m[13]*b->m[4] + a->m[14]*b->m[8] + a->m[15]*b->m[12];
m->m[13] = a->m[12]*b->m[1] + a->m[13]*b->m[5] + a->m[14]*b->m[9] + a->m[15]*b->m[13];
m->m[14] = a->m[12]*b->m[2] + a->m[13]*b->m[6] + a->m[14]*b->m[10] + a->m[15]*b->m[14];
m->m[15] = a->m[12]*b->m[3] + a->m[13]*b->m[7] + a->m[14]*b->m[11] + a->m[15]*b->m[15];
}

float Math3D::Vec3Dot( const VECTOR3 pV1 , const VECTOR3 pV2)
{
    return pV1.x*pV2.x + pV1.y*pV2.y + pV1.z*pV2.z;
}

VECTOR3 Math3D::decreaceVector(VECTOR3 value0 , VECTOR3 value1)
{
VECTOR3 rv;

rv.x = value0.x - value1.x;
rv.y = value0.y - value1.y;
rv.z = value0.z - value1.z;

return rv;
}

VECTOR3 Math3D::Vec3Cross(const VECTOR3 pV1 , const VECTOR3 pV2)
{
VECTOR3 v;

v.x = pV1.y * pV2.z - pV1.z * pV2.y;
v.y = pV1.z * pV2.x - pV1.x * pV2.z;
v.z = pV1.x * pV2.y - pV1.y * pV2.x;

return v; // [Math3D Vec3Normalize:v];
}

VECTOR3 Math3D::Vec3Normalize(const VECTOR3 value)
{
    if (value.x*value.x + value.y*value.y + value.z*value.z == 0.0f)
    {
        return value;
    }

    float fLength = sqrtf( value.x*value.x + value.y*value.y + value.z*value.z );
    VECTOR3 rv;



    if (value.x == 0.0f) {
        rv.x = 0.0;
    }
    else {
        rv.x = value.x/fLength;
    }

    if (value.y == 0.0f) {
        rv.y = 0.0;
    }
    else {
        rv.y = value.y/fLength;
    }


    if (value.z == 0.0f) {
        rv.z = 0.0;
    }
    else {
        rv.z = value.z/fLength;
    }



    return rv;
}

float Math3D::determinate(const Matrix4x4 *m)
{
return -getMatrixValue(m, 0, 0) * getMatrixValue(m, 1, 3) * getMatrixValue(m, 2, 2) * getMatrixValue(m, 3, 1)
+getMatrixValue(m, 0, 0) * getMatrixValue(m, 1, 2) * getMatrixValue(m, 2, 3) * getMatrixValue(m, 3, 1)
+getMatrixValue(m, 0, 0) * getMatrixValue(m, 1, 3) * getMatrixValue(m, 2, 1) * getMatrixValue(m, 3, 2)
-getMatrixValue(m, 0, 0) * getMatrixValue(m, 1, 1) * getMatrixValue(m, 2, 3) * getMatrixValue(m, 3, 2)
-getMatrixValue(m, 0, 0) * getMatrixValue(m, 1, 2) * getMatrixValue(m, 2, 1) * getMatrixValue(m, 3, 3)
+getMatrixValue(m, 0, 0) * getMatrixValue(m, 1, 1) * getMatrixValue(m, 2, 2) * getMatrixValue(m, 3, 3)

-getMatrixValue(m, 0, 3) * getMatrixValue(m, 1, 0) * getMatrixValue(m, 2, 1) * getMatrixValue(m, 3, 2)
+getMatrixValue(m, 0, 3) * getMatrixValue(m, 1, 0) * getMatrixValue(m, 2, 2) * getMatrixValue(m, 3, 1)
+getMatrixValue(m, 0, 3) * getMatrixValue(m, 1, 1) * getMatrixValue(m, 2, 0) * getMatrixValue(m, 3, 2)
+getMatrixValue(m, 0, 3) * getMatrixValue(m, 1, 2) * getMatrixValue(m, 2, 1) * getMatrixValue(m, 3, 0)
-getMatrixValue(m, 0, 3) * getMatrixValue(m, 1, 1) * getMatrixValue(m, 2, 2) * getMatrixValue(m, 3, 0)
-getMatrixValue(m, 0, 3) * getMatrixValue(m, 1, 2) * getMatrixValue(m, 2, 0) * getMatrixValue(m, 3, 1)

+getMatrixValue(m, 0, 2) * getMatrixValue(m, 1, 0) * getMatrixValue(m, 2, 1) * getMatrixValue(m, 3, 3)
-getMatrixValue(m, 0, 2) * getMatrixValue(m, 1, 0) * getMatrixValue(m, 2, 3) * getMatrixValue(m, 3, 1)
-getMatrixValue(m, 0, 2) * getMatrixValue(m, 1, 1) * getMatrixValue(m, 2, 0) * getMatrixValue(m, 3, 3)
+getMatrixValue(m, 0, 2) * getMatrixValue(m, 1, 1) * getMatrixValue(m, 2, 3) * getMatrixValue(m, 3, 0)
+getMatrixValue(m, 0, 2) * getMatrixValue(m, 1, 3) * getMatrixValue(m, 2, 0) * getMatrixValue(m, 3, 1)
-getMatrixValue(m, 0, 2) * getMatrixValue(m, 1, 3) * getMatrixValue(m, 2, 1) * getMatrixValue(m, 3, 0)

+getMatrixValue(m, 0, 1) * getMatrixValue(m, 1, 3) * getMatrixValue(m, 2, 2) * getMatrixValue(m, 3, 0)
-getMatrixValue(m, 0, 1) * getMatrixValue(m, 1, 2) * getMatrixValue(m, 2, 3) * getMatrixValue(m, 3, 0)
-getMatrixValue(m, 0, 1) * getMatrixValue(m, 1, 3) * getMatrixValue(m, 2, 0) * getMatrixValue(m, 3, 2)
+getMatrixValue(m, 0, 1) * getMatrixValue(m, 1, 0) * getMatrixValue(m, 2, 3) * getMatrixValue(m, 3, 2)
+getMatrixValue(m, 0, 1) * getMatrixValue(m, 1, 2) * getMatrixValue(m, 2, 0) * getMatrixValue(m, 3, 3)
-getMatrixValue(m, 0, 1) * getMatrixValue(m, 1, 0) * getMatrixValue(m, 2, 2) * getMatrixValue(m, 3, 3);


}

VECTOR3 Math3D::vectorMatrix(const VECTOR3 * vector , const Matrix4x4 *m)
{
VECTOR3 rv;

rv.x = getMatrixValue(m, 0, 0)*vector->x + getMatrixValue(m, 0, 1)*vector->y + getMatrixValue(m, 0, 2)*vector->z;
rv.y = getMatrixValue(m, 1, 0)*vector->x + getMatrixValue(m, 1, 1)*vector->y + getMatrixValue(m, 1, 2)*vector->z;
rv.z = getMatrixValue(m, 2, 0)*vector->x + getMatrixValue(m, 2, 1)*vector->y + getMatrixValue(m, 2, 2)*vector->z;

return rv;
}




void  Math3D::inversion(const Matrix4x4 *inputMatrix , Matrix4x4 *dst)
{
float	tmp[12]; float	src[16]; float det;
/* transpose matrix */

for (int i = 0; i < 4; i++)
{
src[i] = inputMatrix->m[i*4];
src[i+4] = inputMatrix->m[i*4+1];
src[i+8] = inputMatrix->m[i*4+2];
src[i+12] = inputMatrix->m[i*4+3];
}



/* calculate pairs for first 8 elements (cofactors) */
tmp[0] = src[10] * src[15];
tmp[1] = src[11] * src[14];
tmp[2] = src[9] * src[15];
tmp[3] = src[11] * src[13];
tmp[4] = src[9] * src[14];
tmp[5] = src[10] * src[13];
tmp[6] = src[8] * src[15];
tmp[7] = src[11] * src[12];
tmp[8] = src[8] * src[14];
tmp[9] = src[10] * src[12];
tmp[10] = src[8] * src[13];
tmp[11] = src[9] * src[12];
/* calculate first 8 elements (cofactors) */
dst->m[0] = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
dst->m[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
dst->m[1] = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
dst->m[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
dst->m[2] = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
dst->m[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
dst->m[3] = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
dst->m[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
dst->m[4] = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
dst->m[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
dst->m[5] = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
dst->m[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
dst->m[6] = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
dst->m[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
dst->m[7] = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
dst->m[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];
/* calculate pairs for second 8 elements (cofactors) */

tmp[0] = src[2]*src[7];
tmp[1] = src[3]*src[6];
tmp[2] = src[1]*src[7];
tmp[3] = src[3]*src[5];
tmp[4] = src[1]*src[6];
tmp[5] = src[2]*src[5];
/* temp array for pairs	*/ /* array of transpose source matrix */ /* determinant	*/

tmp[6] = src[0]*src[7];
tmp[7] = src[3]*src[4];
tmp[8] = src[0]*src[6];
tmp[9] = src[2]*src[4];
tmp[10] = src[0]*src[5];
tmp[11] = src[1]*src[4];
/* calculate second 8 elements (cofactors) */
dst->m[8] = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
dst->m[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
dst->m[9] = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
dst->m[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
dst->m[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
dst->m[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
dst->m[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
dst->m[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
dst->m[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
dst->m[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
dst->m[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
dst->m[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
dst->m[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
dst->m[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
dst->m[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
dst->m[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];
/* calculate determinant */
det=src[0]*dst->m[0]+src[1]*dst->m[1]+src[2]*dst->m[2]+src[3]*dst->m[3];

det = 1/det;
for (int j = 0; j < 16; j++)
{
dst->m[j] *= det;
}
}

void Math3D::MatrixLookAtLH(Matrix4x4 *outMatrix , const VECTOR3 pEye , const VECTOR3 pAt , const VECTOR3 pUp)
{
    VECTOR3 dec1 = decreaceVector(pAt, pEye);
    VECTOR3 zaxis = Vec3Normalize(dec1);
    VECTOR3 dec2 = Vec3Cross(pUp, zaxis);
    VECTOR3 xaxis = Vec3Normalize(dec2);
    VECTOR3 yaxis = Vec3Cross(zaxis, xaxis);

    setMatrixValue(outMatrix,0,0,xaxis.x);
    setMatrixValue(outMatrix,0,1,yaxis.x);
    setMatrixValue(outMatrix,0,2,zaxis.x);
    setMatrixValue(outMatrix,0,3,0.0f);

    setMatrixValue(outMatrix,1,0,xaxis.y);
    setMatrixValue(outMatrix,1,1,yaxis.y);
    setMatrixValue(outMatrix,1,2,zaxis.y);
    setMatrixValue(outMatrix,1,3,0.0f);


    setMatrixValue(outMatrix,2,0,xaxis.z);
    setMatrixValue(outMatrix,2,1,yaxis.z);
    setMatrixValue(outMatrix,2,2,zaxis.z);
    setMatrixValue(outMatrix,2,3,0.0f);


    setMatrixValue(outMatrix,3,0,-Vec3Dot(xaxis,pEye) );
    setMatrixValue(outMatrix,3,1,-Vec3Dot(yaxis,pEye) );
    setMatrixValue(outMatrix,3,2,-Vec3Dot(zaxis,pEye) );
    setMatrixValue(outMatrix,3,3,1.0f );

}

//void CMain3D::pick()

void Math3D::pick(const CGPoint ptCursor , const int iWidth , const int iHeight , const Matrix4x4 *proj , const Matrix4x4 *view , VECTOR3 *vDir , VECTOR3 *vOrig)
{
VECTOR3 v;
Matrix4x4 dst;

v.x =  ( ( ( 2.0f * ptCursor.x ) / (const float)iWidth  ) - 1 )/getMatrixValue(proj,0,0);
v.y = -( ( ( 2.0f * ptCursor.y ) / (const float)iHeight ) - 1 )/getMatrixValue(proj,1,1);
v.z =  1.0f;

inversion(view, &dst);

vDir->x  = v.x*getMatrixValue(&dst,0,0) + v.y*getMatrixValue(&dst,1,0) + v.z*getMatrixValue(&dst,2,0);
vDir->y  = v.x*getMatrixValue(&dst,0,1) + v.y*getMatrixValue(&dst,1,1) + v.z*getMatrixValue(&dst,2,1);
vDir->z  = v.x*getMatrixValue(&dst,0,2) + v.y*getMatrixValue(&dst,1,2) + v.z*getMatrixValue(&dst,2,2);
vOrig->x = getMatrixValue(&dst,3,0);
vOrig->y = getMatrixValue(&dst,3,1);
vOrig->z = getMatrixValue(&dst,3,2);

}

bool Math3D::doesThisHitToTriangle(const VECTOR3 orig , const VECTOR3 dir , const VECTOR3 v0 , const VECTOR3 v1 , const VECTOR3 v2 , float *t , float * u , float *v , VECTOR3 *outTg , float *distance)
{
// Find vectors for two edges sharing vert0
VECTOR3 edge1 = decreaceVector(v1,v0);
VECTOR3 edge2 = decreaceVector(v2,v0);
float U,V,  T;



// Begin calculating determinant - also used to calculate U parameter
VECTOR3 pvec = Vec3Cross(dir, edge2);


// If determinant is near zero, ray lies in plane of triangle
float det = Vec3Dot(edge1, pvec);



if( det < 0.0001f )
return false;

// Calculate distance from vert0 to ray origin
VECTOR3 tvec = decreaceVector(orig,v0);

// Calculate U parameter and test bounds
U = Vec3Dot(tvec,pvec);
if( U < 0.0f || U > det )
return false;

// Prepare to test V parameter
VECTOR3 qvec = Vec3Cross(tvec,edge1);

// Calculate V parameter and test bounds
V = Vec3Dot(dir,qvec);
if( V < 0.0f || U + V > det )
return false;

// Calculate t, scale parameters, ray intersects triangle
T = Vec3Dot(edge2,qvec);
float fInvDet = 1.0f / det;
T *= fInvDet;
U *= fInvDet;
V *= fInvDet;

*t = T;
*u = U;
*v = V;

float x1 = T*dir.x + orig.x;
float y1 = T*dir.y + orig.y;
float z1 = T*dir.z + orig.z;

outTg->x = T*dir.x + orig.x;
outTg->y = T*dir.y + orig.y;
outTg->z = T*dir.z + orig.z;
*distance = sqrtf(
        (orig.x - x1)*(orig.x - x1) +
        (orig.y - y1)*(orig.y - y1) +
        (orig.z - z1)*(orig.z - z1) );


return true;
}

void Math3D::MatrixPerspectiveFovLH(Matrix4x4 *pOut , float  fovy , float  Aspect , float  zn , float  zf)
{
    float fVal = fovy/2;
    float yScale = 1/tanf(fVal);
    float xScale = yScale / Aspect;

    setMatrixValue(pOut,0,0,xScale);
    setMatrixValue(pOut,0,1,0);
    setMatrixValue(pOut,0,2,0);
    setMatrixValue(pOut,0,3,0);

    setMatrixValue(pOut,1,0,0);
    setMatrixValue(pOut,1,1,yScale);
    setMatrixValue(pOut,1,2,0);
    setMatrixValue(pOut,1,3,0);

    setMatrixValue(pOut,2,0,0);
    setMatrixValue(pOut,2,1,0);
    setMatrixValue(pOut,2,2,zf/(zf-zn) );
    setMatrixValue(pOut,2,3,1);

    setMatrixValue(pOut,3,0,0);
    setMatrixValue(pOut,3,1,0);
    setMatrixValue(pOut,3,2,-zn*zf/(zf-zn));
    setMatrixValue(pOut,3,3,0);
}

void Math3D::SetOrtho(Matrix4x4 *m , float  left , float  right , float  bottom , float  top , float  near , float  far)
{
const float tx = - (right + left)/(right - left);
const float ty = - (top + bottom)/(top - bottom);
const float tz = - (far + near)/(far - near);

m->m[0] = 2.0f/(right-left);
m->m[1] = 0;
m->m[2] = 0;
m->m[3] = tx;

m->m[4] = 0;
m->m[5] = 2.0f/(top-bottom);
m->m[6] = 0;
m->m[7] = ty;

m->m[8] = 0;
m->m[9] = 0;
m->m[10] = -2.0/(far-near);
m->m[11] = tz;

m->m[12] = 0;
m->m[13] = 0;
m->m[14] = 0;
m->m[15] = 1;
}

void Math3D::MatrixTranslation(Matrix4x4 *outMatrix , float  x , float  y , float  z)
{
setMatrixValue(outMatrix,0,0,1.0f);
setMatrixValue(outMatrix,0,1,0.0f);
setMatrixValue(outMatrix,0,2,0.0f);
setMatrixValue(outMatrix,0,3,0.0f);

setMatrixValue(outMatrix,1,0,0.0f);
setMatrixValue(outMatrix,1,1,1.0f);
setMatrixValue(outMatrix,1,2,0.0f);
setMatrixValue(outMatrix,1,3,0.0f);


setMatrixValue(outMatrix,2,0,0.0f);
setMatrixValue(outMatrix,2,1,0.0f);
setMatrixValue(outMatrix,2,2,1.0f);
setMatrixValue(outMatrix,2,3,0.0f);


setMatrixValue(outMatrix,3,0,x);
setMatrixValue(outMatrix,3,1,y);
setMatrixValue(outMatrix,3,2,z);
setMatrixValue(outMatrix,3,3,1.0f);
}

float Math3D::calcAngleY(const VECTOR3 start , const VECTOR3 end)
{
float distanceXZ;
float dx, dy, dz, rv;

dx = end.x - start.x;
dy = end.y - start.y;
dz = end.z - start.z;

if (dx*dx + dz*dz == 0)
{
if (dy < 0)
{
return -PI_VALUE/2;
}
return PI_VALUE/2;
}

distanceXZ = sqrtf(dx*dx + dz*dz);
rv = (float)atan(dy/distanceXZ);

return rv;
}

float Math3D::calcAngleY( const VECTOR3 end)
{
float distanceXZ;

if (end.x*end.x + end.z*end.z == 0)
{
if (end.y < 0)
{
return -PI_VALUE/2;
}
return PI_VALUE/2;
}

distanceXZ = sqrtf(end.x*end.x + end.z*end.z);

return (float)atan(end.y/distanceXZ);
}

float Math3D::getYFromAngleY(float  x , float  z , const float fAngleY)
{
if (fAngleY == 0 || x*x + z*z == 0) {

return 0;
}

float distanceXZ = sqrtf(x*x + z*z);

return (float)tan(fAngleY)*distanceXZ;
}

float Math3D::calcAngleXZ( const VECTOR3 start , const VECTOR3 end)
{

if (start.x == end.x)
{
if (start.z <= end.z)
{
return 0;
}

return PI_VALUE;
}

if (start.z == end.z)
{
if (start.x < end.x)
{
return PI_VALUE/2;
}
return -PI_VALUE/2;
}


float rv = (end.x - start.x)/(end.z - start.z);




if (rv < 0)
{
rv *= -1.0f;
}

rv = (float)atanf(rv);

if (start.z <= end.z && start.x <= end.x )
{
return rv;
}
if (start.z >= end.z && start.x <= end.x )
{
return PI_VALUE - rv;
}


if (start.z <= end.z && start.x >= end.x )
{
return -rv;
}

if (start.z >= end.z && start.x >= end.x )
{
return -(PI_VALUE - rv);
}

return rv;
}

float Math3D::calcAngleXZ( const VECTOR3 end)
{

if (0 == end.x)
{
if (0 <= end.z)
{
return 0;
}

return PI_VALUE;
}

if (0 == end.z)
{
if (0 < end.x)
{
return PI_VALUE/2;
}
return -PI_VALUE/2;
}


float rv = (end.x)/(end.z);

if (rv < 0)
{
rv *= -1.0f;
}

rv = (float)atanf(rv);

if (0 <= end.z && 0 <= end.x )
{
return rv;
}
if (0 >= end.z && 0 <= end.x )
{
return PI_VALUE - rv;
}


if (0 <= end.z && 0 >= end.x )
{
return -rv;
}

if (0 >= end.z && 0 >= end.x )
{
return -(PI_VALUE - rv);
}

return rv;
}

float Math3D::calculateAngle( float  x , float  y)
{
if (x == 0)
{
if (y >= 0)
{
return 0;
}

return M_PI;
}

if (y == 0)
{
if (x >= 0)
{
return M_PI/2.0f;
}

return M_PI + M_PI/2.0f;
}

if (x >= 0 && y >= 0)
{
return atanf(x/y);
}

if (x >= 0 && y <= 0)
{
return M_PI + atanf(x/y);
}

if (x <= 0 && y <= 0)
{
return M_PI + atanf(x/y);
}

return M_PI*2 + atanf(x/y);
}

void Math3D::setAngleXZ(float *rvX , float *rvZ , float  fAngle)
{
fAngle = fixAngle(fAngle);
// if (PI_VALUE/2.0f >= fAngle && fAngle >= -PI_VALUE)
{
*rvX = sinf(fAngle);
*rvZ = cosf(fAngle);
}

/*else
{
    *rvX = cosf(fAngle);
    *rvZ = sinf(fAngle);
}
 */
}


void Math3D::MatrixOrthoLH( Matrix4x4 *pOut , float  w , float  h , float  zn , float  zf)
{
setMatrixValue(pOut,0,0,2.0f/w);
setMatrixValue(pOut,0,1,0);
setMatrixValue(pOut,0,2,0);
setMatrixValue(pOut,0,3,0);

setMatrixValue(pOut,1,0,0);
setMatrixValue(pOut,1,1,2.0f/h);
setMatrixValue(pOut,1,2,0);
setMatrixValue(pOut,1,3,0);

setMatrixValue(pOut,2,0,0);
setMatrixValue(pOut,2,1,0);
setMatrixValue(pOut,2,2,1/(zf-zn) );
setMatrixValue(pOut,2,3,0);

setMatrixValue(pOut,3,0,0);
setMatrixValue(pOut,3,1,0);
setMatrixValue(pOut,3,2,zn/(zn-zf));
setMatrixValue(pOut,3,3,1);
}
/*							   __inout  D3DXMATRIX *pOut,
							   __in     FLOAT w,
							   __in     FLOAT h,
							   __in     FLOAT zn,
							   __in     FLOAT zf
							   )

 2/w  0    0           0
 0    2/h  0           0
 0    0    1/(zf-zn)   0
 0    0    zn/(zn-zf)  1
 */
void Math3D::SetRotateY( Matrix4x4 *m , float  angle)
{
const float c = cosf(angle);
const float s = sinf(angle);

setMatrixValue(m,0,0,c);
setMatrixValue(m,0,1,0);
setMatrixValue(m,0,2,s);
setMatrixValue(m,0,3,0);

setMatrixValue(m,1,0,0);
setMatrixValue(m,1,1,1);
setMatrixValue(m,1,2,0);
setMatrixValue(m,1,3,0);

setMatrixValue(m,2,0,-s);
setMatrixValue(m,2,1,0);
setMatrixValue(m,2,2,c);
setMatrixValue(m,2,3,0);

setMatrixValue(m,3,0,0);
setMatrixValue(m,3,1,0);
setMatrixValue(m,3,2,0);
setMatrixValue(m,3,3,1);
}

void Math3D::SetRotateX( Matrix4x4 *m , float  angle)
{
const float c = cosf(angle);
const float s = sinf(angle);

setMatrixValue(m,0,0,1);
setMatrixValue(m,0,1,0);
setMatrixValue(m,0,2,0);
setMatrixValue(m,0,3,0);

setMatrixValue(m,1,0,0);
setMatrixValue(m,1,1,c);
setMatrixValue(m,1,2,-s);
setMatrixValue(m,1,3,0);

setMatrixValue(m,2,0,0);
setMatrixValue(m,2,1,s);
setMatrixValue(m,2,2,c);
setMatrixValue(m,2,3,0);

setMatrixValue(m,3,0,0);
setMatrixValue(m,3,1,0);
setMatrixValue(m,3,2,0);
setMatrixValue(m,3,3,1);
}

void Math3D::SetRotateZ( Matrix4x4 *m , float  angle)
{
const float c = cosf(angle);
const float s = sinf(angle);

setMatrixValue(m,0,0,c);
setMatrixValue(m,0,1,-s);
setMatrixValue(m,0,2,0);
setMatrixValue(m,0,3,0);

setMatrixValue(m,1,0,s);
setMatrixValue(m,1,1,c);
setMatrixValue(m,1,2,0);
setMatrixValue(m,1,3,0);

setMatrixValue(m,2,0,0);
setMatrixValue(m,2,1,0);
setMatrixValue(m,2,2,1);
setMatrixValue(m,2,3,0);

setMatrixValue(m,3,0,0);
setMatrixValue(m,3,1,0);
setMatrixValue(m,3,2,0);
setMatrixValue(m,3,3,1);
}

void Math3D::SetRotateZYX( Matrix4x4 *m , float  angleX , float  angleY , float  angleZ)
{
const float cX = cosf(angleX);
const float sX = sinf(angleX);
const float cY = cosf(angleY);
const float sY = sinf(angleY);
const float cZ = cosf(angleZ);
const float sZ = sinf(angleZ);

setMatrixValue(m,0,0,cZ*cY);
setMatrixValue(m,0,1,cZ*sY*sZ - sZ*cX);
setMatrixValue(m,0,2,cZ*sY*cX + sZ*sX);
setMatrixValue(m,0,3,0);

setMatrixValue(m,1,0,sZ*cY);
setMatrixValue(m,1,1,sZ*sY*sX + cZ*cX);
setMatrixValue(m,1,2,sZ*sY*cX - cZ*sX);
setMatrixValue(m,1,3,0);

setMatrixValue(m,2,0,-sY);
setMatrixValue(m,2,1,cY*sX);
setMatrixValue(m,2,2,cY*cX);
setMatrixValue(m,2,3,0);

setMatrixValue(m,3,0,0);
setMatrixValue(m,3,1,0);
setMatrixValue(m,3,2,0);
setMatrixValue(m,3,3,1.0f);

}


void Math3D::SetRotateXZY(Matrix4x4 *m , float  angleX , float  angleY , float  angleZ)
{
const float c1 = cosf(angleX);
const float s1 = sinf(angleX);
const float c3 = cosf(angleY);
const float s3 = sinf(angleY);
const float c2 = cosf(angleZ);
const float s2 = sinf(angleZ);

setMatrixValue(m,0,0,c2*c3);
setMatrixValue(m,0,1,-s2);
setMatrixValue(m,0,2,c2*s3);
setMatrixValue(m,0,3,0);

setMatrixValue(m,1,0,s1*s3+c3*c1*s2);
setMatrixValue(m,1,1,c1*c2);
setMatrixValue(m,1,2,c1*s2*s3-c3*s1);
setMatrixValue(m,1,3,0);

setMatrixValue(m,2,0,c3*s1*s2-c1*s3);
setMatrixValue(m,2,1,c2*s1);
setMatrixValue(m,2,2,s1*s2*s3+c1*c3);
setMatrixValue(m,2,3,0);

setMatrixValue(m,3,0,0);
setMatrixValue(m,3,1,0);
setMatrixValue(m,3,2,0);
setMatrixValue(m,3,3,1.0f);

}

void Math3D::SetRotateYXZ( Matrix4x4 *m , float  angleX , float  angleY , float  angleZ)
{
const float c2 = cosf(angleX);
const float s2 = sinf(angleX);
const float c1 = cosf(angleY);
const float s1 = sinf(angleY);
const float c3 = cosf(angleZ);
const float s3 = sinf(angleZ);

setMatrixValue(m,0,0,c1*c3 + s1*s2*s3);
setMatrixValue(m,0,1,c3*s1*s2-c1*s3);
setMatrixValue(m,0,2,c2*s1);
setMatrixValue(m,0,3,0);

setMatrixValue(m,1,0,c2*s3);
setMatrixValue(m,1,1,c2*c3);
setMatrixValue(m,1,2,-s2);
setMatrixValue(m,1,3,0);

setMatrixValue(m,2,0,c1*s2*s3-c3*s1);
setMatrixValue(m,2,1,s1*s3+c1*c3*s2);
setMatrixValue(m,2,2,c1*c2);
setMatrixValue(m,2,3,0);

setMatrixValue(m,3,0,0);
setMatrixValue(m,3,1,0);
setMatrixValue(m,3,2,0);
setMatrixValue(m,3,3,1.0f);

}

VECTOR3 Math3D::calculateAroundBallAngle(const VECTOR3 vector , float  angle)
{
VECTOR3 rv;
float fAngleY;
VECTOR3 s;
s.x = 0;
s.y = 0;
s.z = 0;

fAngleY = calcAngleY(s,vector);

float fNewAngleY = angle + fAngleY;

if (angle >= 0)
{
if (fNewAngleY >= PI_VALUE/2.0f - 0.05f)
{
fNewAngleY = PI_VALUE/2.0f - 0.05f;
}
}
else
{
if (fNewAngleY <= -PI_VALUE/2.0f + 0.05f)
{
fNewAngleY = -PI_VALUE/2.0f + 0.05f;
}
}

float dx = vector.x;
float dy = vector.y;
float dz = vector.z;
float distanceXZ = sqrtf(dx*dx + dz*dz);
float distanceOld = sqrtf(dx*dx + dy*dy + dz*dz);

dy = tan(fNewAngleY)*distanceXZ;

rv.x = vector.x;
rv.y = dy;
rv.z = vector.z;

rv = moveDistance2(s,rv,distanceOld);

return rv;
}

float Math3D::getDistance(float  x0 , float  y0 , float  x1 , float  y1)
{
return sqrtf( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) );
}

float Math3D::getDistance3D(float  x0 , float  y0 , float  z0 , float  x1 , float  y1 , float  z1)
{
return sqrtf( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0) );
}

/*
 + (float)calcAngleY , const VECTOR3 start , const VECTOR3 end;
 + (float)calcAngleXZ , const VECTOR3 start , const VECTOR3 end;

+(void)SetRotateXYZ , Matrix4x4 *result :(GLfloat)angle :(GLfloat)x :(GLfloat)y :(GLfloat)z
{
	GLfloat sinAngle, cosAngle;
	GLfloat mag = sqrtf(x * x + y * y + z * z);

	sinAngle = sinf ( angle * PI / 180.0f );
	cosAngle = cosf ( angle * PI / 180.0f );
	if ( mag > 0.0f )
	{
		GLfloat xx, yy, zz, xy, yz, zx, xs, ys, zs;
		GLfloat oneMinusCos;
		ESMatrix rotMat;

		x /= mag;
		y /= mag;
		z /= mag;

		xx = x * x;
		yy = y * y;
		zz = z * z;
		xy = x * y;
		yz = y * z;
		zx = z * x;
		xs = x * sinAngle;
		ys = y * sinAngle;
		zs = z * sinAngle;
		oneMinusCos = 1.0f - cosAngle;

		setMatrixValue(result:0:0 :(oneMinusCos * xx) + cosAngle];
		setMatrixValue(result:1:0 :(oneMinusCos * xy) - zs];
		setMatrixValue(result:2:0 :(oneMinusCos * zx) + ys];
		setMatrixValue(result:3:0:0.0f];

		setMatrixValue(result:0:1 :(oneMinusCos * xy) + zs];
		setMatrixValue(result:1:1 :(oneMinusCos * yy) + cosAngle];
		setMatrixValue(result:2:1 :(oneMinusCos * yz) - xs];
		setMatrixValue(result:3:1:0.0f];

		setMatrixValue(result:0:2 :(oneMinusCos * zx) - ys];
		setMatrixValue(result:1:2 :(oneMinusCos * yz) + xs];
		setMatrixValue(result:2:2 :(oneMinusCos * zz) + cosAngle];
		setMatrixValue(result:3:2:0.0f];

		setMatrixValue(result:0:3:0.0f];
		setMatrixValue(result:1:3:0.0f];
		setMatrixValue(result:2:3:0.0f];
		setMatrixValue(result:3:3:1.0f];



	}
}
*/

VECTOR3 Math3D::calculatePointFromLineByOrigDistance(const VECTOR3 vLinePoint , const VECTOR3 vLineDir , float  vLineStartDistance , const VECTOR3 vOrig , const float vOrigWantDistance)
{
int i;
VECTOR3 rv;
VECTOR3 vTempVector;
float fRvDistance;
float fTempValue[2];
float fTempDistance;
float fDiffX;
float fDiffY;
float fDiffZ;
bool bEnlarge = false;
float fStep;
float fNewLineStartDistance;

rv = moveDistance(vLinePoint,vLineDir,vLineStartDistance);
fDiffX = rv.x - vOrig.x;
fDiffY = rv.y - vOrig.y;
fDiffZ = rv.z - vOrig.z;
fRvDistance = sqrtf(fDiffX*fDiffX + fDiffY*fDiffY + fDiffZ*fDiffZ);

fStep = (vOrigWantDistance - fRvDistance)/10.0f;
if (fRvDistance < vOrigWantDistance)
{
bEnlarge = true;
}
else
{
bEnlarge = false;
}

if (fStep < 0)
{
fStep = -fStep;
}

for (i=0;i<100;i++)
{
if (bEnlarge)
{
fNewLineStartDistance = vLineStartDistance + fStep;
}
else
{
fNewLineStartDistance = vLineStartDistance - fStep;
}

vTempVector = Math3D::moveDistance(vLinePoint,vLineDir, fNewLineStartDistance);

fDiffX = vTempVector.x - vOrig.x;
fDiffY = vTempVector.y - vOrig.y;
fDiffZ = vTempVector.z - vOrig.z;
fTempDistance = sqrtf(fDiffX*fDiffX + fDiffY*fDiffY + fDiffZ*fDiffZ);

if (fTempDistance == vOrigWantDistance)
{
return vTempVector;
}

fTempValue[0] = fRvDistance - vOrigWantDistance;
fTempValue[1] = fTempDistance - vOrigWantDistance;

if (fTempValue[0] < 0)
{
fTempValue[0] = -fTempValue[0];
}

if (fTempValue[1] < 0)
{
fTempValue[1] = -fTempValue[1];
}

if (fTempValue[1] > fTempValue[0])
{
fStep /= 2.0f;
}
else
{
vLineStartDistance = fNewLineStartDistance;
rv = vTempVector;

fDiffX = rv.x - vOrig.x;
fDiffY = rv.y - vOrig.y;
fDiffZ = rv.z - vOrig.z;
fRvDistance = sqrtf(fDiffX*fDiffX + fDiffY*fDiffY + fDiffZ*fDiffZ);

if (fRvDistance < vOrigWantDistance)
{
bEnlarge = true;
}
else
{
bEnlarge = false;
}
}
}

return rv;
}

VECTOR3 Math3D::moveDistance( const VECTOR3 pOrig , const VECTOR3 vDir , float  fDistance)
{
VECTOR3 rv;

if (vDir.x*vDir.x + vDir.y*vDir.y + vDir.z*vDir.z == 0)
{
return pOrig;
}

float distanceRate = sqrtf(fDistance*fDistance/(vDir.x*vDir.x + vDir.y*vDir.y + vDir.z*vDir.z));

rv.x = pOrig.x + vDir.x*distanceRate;
rv.y = pOrig.y + vDir.y*distanceRate;
rv.z = pOrig.z + vDir.z*distanceRate;

return rv;
}

float Math3D::calculateAngleDifference(float  x1 , float  y1 , float  x2 , float  y2)
{
float fAngle1 = Math3D::calculateAngle(x1,y1);
float fAngle2 = Math3D::calculateAngle(x2,y2);

return fixAngle(fAngle2 - fAngle1);
}

float Math3D::calculateClosestAngleDifference(float  fAngle0 , float  fAngle1)
{
float fDifference0 = fixAngle(fAngle0 - fAngle1);

if (fDifference0 > PI_VALUE)
{
fDifference0 = PI_VALUE*2 - fDifference0;
}

return fDifference0;
}

bool Math3D::isAngleBetween(float  fFromAngle , float  fToAngle , float  fAngleBetween)
{
// fFromAngle = Math3D::fixAngle(fFromAngle];
//  fToAngle = Math3D::fixAngle(fToAngle];
// fAngleBetween = Math3D::fixAngle(fAngleBetween];

if (fToAngle > fFromAngle)
{
if (fToAngle >= fAngleBetween && fFromAngle <= fAngleBetween)
{
return true;
}
}
else
{
if (fAngleBetween >= fToAngle || fAngleBetween <= fFromAngle)
{
return true;
}
}

return false;
}

float Math3D::getGravityTanget(float  fGravity)
{
return atanf(fGravity)/800.0f;
}

float Math3D::fixAngle(float  fAngle)
{
while (1)
{
if (fAngle >= 0 && fAngle < M_PI*2)
{
return fAngle;
}

if (fAngle < 0)
{
fAngle += M_PI*2;
}
else {
fAngle -= M_PI*2;
}
}
}

bool Math3D::isAngleForward( float  fAngle , float  fToAngle)
{
int i;

fAngle = Math3D::fixAngle(fAngle);
fToAngle = Math3D::fixAngle(fToAngle);

for (i=0;i<2;i++)
{
if (fAngle <= fToAngle && fAngle + M_PI >= fToAngle)
{
return true;
}

fAngle += M_PI;
fToAngle += M_PI;

fAngle = Math3D::fixAngle(fAngle);
fToAngle = Math3D::fixAngle(fToAngle);
}

return false;

}

bool Math3D::isAngleForward( float  x , float  y , float  fToAngle)
{

float fFromAngle = Math3D::calculateAngle(x,y);
return isAngleForward(fFromAngle,fToAngle);
}


void Math3D::rotateByAngle(float *fOutX , float *fOutY, float x , float y, float fAngle)
{
    float fDistance;
    if (std::abs(x) <= 0 && std::abs(y) <= 0) {
        *fOutX = x;
        *fOutY = y;
        return;
    }

    fAngle += calculateAngle(x, y);
    fDistance = sqrtf(x*x + y*y);
    fAngle = Math3D::fixAngle(fAngle);

    *fOutX = sinf(fAngle)*fDistance;
    *fOutY = cosf(fAngle)*fDistance;
}

VECTOR3 Math3D::moveDistance2( const VECTOR3 vStart , const VECTOR3 vEnd , float  fDistance)
{
VECTOR3 rv;
float fDiffX = vEnd.x - vStart.x;
float fDiffY = vEnd.y - vStart.y;
float fDiffZ = vEnd.z - vStart.z;

if (fDiffX*fDiffX + fDiffY*fDiffY + fDiffZ*fDiffZ==0)
{
return vStart;
}

float fFullDistance = sqrtf(fDiffX*fDiffX + fDiffY*fDiffY + fDiffZ*fDiffZ);

rv.x = fDiffX*fDistance/fFullDistance + vStart.x;
rv.y = fDiffY*fDistance/fFullDistance + vStart.y;
rv.z = fDiffZ*fDistance/fFullDistance + vStart.z;

return rv;
}

VECTOR2 Math3D::getPositionFromOrigDir(VECTOR3 vOrig , VECTOR3 vDir , float  fHeight)
{
VECTOR2 rv;

float fHeighDistance = (vOrig.y - fHeight)/vDir.y;

rv.x = vOrig.x - fHeighDistance*vDir.x;
rv.y = vOrig.z - fHeighDistance*vDir.z;

return rv;
}


VECTOR3 Math3D::set2Dto3DBall(float  fPixelX , float  fPixelY , float  fMaxSizeX , float  fMaxSizeY)
{
VECTOR3 rv;
if (fPixelX > fMaxSizeX/2.0f)
{
rv = Math3D::set2Dto3DBall(fPixelX-fMaxSizeX/2.0f,fPixelY,fMaxSizeX,fMaxSizeY);
rv.x = -rv.x;
rv.z = -rv.z;
return rv;
}
float fPositionX = 0;
float fPositionY = 0;
float fPositionZ = 0;
float fAngleX;
float fAngleY;
float fDistanceHorizontalFromCenter;
bool bNegativePositionY;
bool bNegativePositionZ;

if (fPixelY <= fMaxSizeY/2.0f)
{
fPixelY = fMaxSizeY/2.0f - fPixelY;
fAngleY = fPixelY*(M_PI/2.0f)/(fMaxSizeY/2.0f);
fPositionY = sinf(fAngleY);
fDistanceHorizontalFromCenter = cosf(fAngleY);
bNegativePositionY = false;

}
else
{
bNegativePositionY = true;
fPixelY = fPixelY - fMaxSizeY/2.0f;
fAngleY = fPixelY*(M_PI/2.0f)/(fMaxSizeY/2.0f);
fPositionY = sinf(fAngleY);
fDistanceHorizontalFromCenter = cosf(fAngleY);
}

if (fPixelX <= fMaxSizeX/2.0f)
{
// fPixelX = fMaxSizeX - fPixelX;
bNegativePositionZ = true;
}
else
{
bNegativePositionZ = false;
}

if (fPixelX <= fMaxSizeX/2.0f)
{
if (fPixelX <= fMaxSizeX/4.0f)
{
//float fTempPixelX =
fAngleX = fPixelX*(M_PI/2.0f)/(fMaxSizeX/4.0f);
fPositionX = -cosf(fAngleX);
if (bNegativePositionZ)
{
fPositionZ = -sinf(fAngleX);
}
else
{
fPositionZ = sinf(fAngleX);
}
}
else
{
fPixelX = fMaxSizeX/2.0f - fPixelX;
fAngleX = fPixelX*(M_PI/2.0f)/(fMaxSizeX/4.0f);
fPositionX = cosf(fAngleX);
if (bNegativePositionZ)
{
fPositionZ = -sinf(fAngleX);
}
else
{
fPositionZ = sinf(fAngleX);
}
}
}

fPositionX *= fDistanceHorizontalFromCenter;
fPositionZ *= fDistanceHorizontalFromCenter;

if (bNegativePositionY)
{
fPositionY *= -1;
}

rv.x = fPositionX;
rv.y = fPositionY;
rv.z = fPositionZ;

return rv;
}
