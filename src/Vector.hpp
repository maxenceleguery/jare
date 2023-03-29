#pragma once
#include <iostream>
#include <cstring>

template<typename T>
class Vector {

    private:
        T x;
        T y;
        T z;

        double fast_inverse_square_root(float number) const {
            uint32_t i;
            float x2, y0;
            const float threehalfs = 1.5F;

            x2 = number*0.5F;
            y0=number;
            memcpy(&i, &y0, 4); // i=*(long*) &y0; //evil floating point bit hack
            i = 0x5f3759df - (i >> 1); //what the fuck ?
            memcpy(&y0, &i, 4); // y0=*(float*) &i;
            y0=y0*(threehalfs - (x2*y0*y0)); // Newton's method
            y0=y0*(threehalfs - (x2*y0*y0)); // Newton's method again
            return (double)y0;
        }
        
    public:
        Vector(T x0, T y0, T z0) : x(x0), y(y0), z(z0) {};
        Vector(Vector& vec) : x(vec.x), y(vec.y), z(vec.z) {};
        ~Vector(){};

        void printCoord() const {
            std::cout << x << " " << y << " " << z << std::endl;
        }

        void normalize() {
            double invNorm = fast_inverse_square_root(x*x + y*y + z*z);
            this->x*=invNorm;
            this->y*=invNorm;
            this->z*=invNorm;
        }

        template<typename U>
        Vector& operator = (const Vector<U>& vec) {
            this->x = static_cast<T>(vec.x);
            this->y = static_cast<T>(vec.y);
            this->z = static_cast<T> (vec.z);
            return *this;
        }

        Vector<T> operator + (const Vector<T>& pos) const {
            return Vector<T>(x+pos.x,y+pos.y,z+pos.z);
        }

        Vector<T> operator- (const Vector<T>& pos) const {
            return Vector<T>(x-pos.x,y-pos.y,z-pos.z);
        }

        Vector<T> operator- () const {
            return Vector<T>(-x,-y,-z);
        }

        Vector<T>& operator+= (const T nb) {
            this->x += nb;
            this->y += nb;
            this->z += nb;
            return *this;
        }

        Vector<T>& operator+= (const Vector<T>& pos) {
            this->x += pos.x;
            this->y += pos.y;
            this->z += pos.z;
            return *this;
        }

        Vector<T>& operator-= (const T nb) {
            this->x -= nb;
            this->y -= nb;
            this->z -= nb;
            return *this;
        }

        Vector<T>& operator-= (const Vector<T>& pos) {
            this->x -= pos.x;
            this->y -= pos.y;
            this->z -= pos.z;
            return *this;
        }

        Vector<T>& operator*= (const T nb) {
            this->x *= nb;
            this->y *= nb;
            this->z *= nb;
            return *this;
        }

        Vector<T>& operator/= (const T nb) {
            this->x /= nb;
            this->y /= nb;
            this->z /= nb;
            return *this;
        }

        T operator * (const Vector<T>& pos) const {
            return x*pos.x + y*pos.y + z*pos.z;
        }

        template <typename U>
        bool operator == (const Vector<U>& pos) const {
            if (std::is_same<T,U>::value) {
                return x==pos.x && y==pos.y && z==pos.z;
            }
            return false;
        }
};
