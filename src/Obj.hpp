#pragma once
#include "Vector.hpp"

#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>

class Obj {
    private:
        std::string nameObj;
        std::vector<Vector<float>> v;
        std::vector<Vector<float>> vt;
        std::vector<Vector<float>> vn;
        std::vector<std::vector<Vector<int>>> trianglesVecticesIndexes;

        std::vector<std::string> split(std::string ligne, char delim) {
            std::vector<std::string> splitted;
            std::istringstream f(ligne);
            std::string s;
            while (getline(f, s, delim)) {
                splitted.push_back(s);
            }
            return splitted;
        }

        std::string& rtrim(std::string &str, std::string const &whitespace = " \r\n\t\v\f") {
            str.erase(str.find_last_not_of(whitespace) + 1);
            return str;
        }

    public:
        Obj(const std::string name) {
            std::string path = std::string("./models/");
            std::ifstream monFlux((path+name).c_str());
            std::string ligne;

            while (getline(monFlux, ligne)) {
                std::vector<std::string> splitted = split(ligne, ' ');
                if (splitted[0] == "o") {
                    nameObj = splitted[1];
                }
                else if (splitted[0] == "v") {
                    float x = atof(splitted[1].c_str());
                    float y = atof(splitted[2].c_str());
                    float z = atof(splitted[3].c_str());
                    v.push_back(Vector<float>(x, y, z));
                }
                else if (splitted[0] == "vt") {
                    float xt = atof(splitted[1].c_str());
                    float yt = atof(splitted[2].c_str());
                    vt.push_back(Vector<float>(xt, yt, 0.));
                }
                else if (splitted[0] == "vn") {
                    float xn = atof(splitted[1].c_str());
                    float yn = atof(splitted[2].c_str());
                    float zn = atof(splitted[3].c_str());
                    vn.push_back(Vector<float>(xn, yn, zn));
                }
                else if (splitted[0] == "f") {
                    splitted = split(rtrim(ligne), ' ');
                    std::vector<Vector<int>> vecticesIndexes;
                    for (uint i=1;i<splitted.size();i++) {
                        std::vector<std::string> splittedTriangles = split(splitted[i], '/');
                        vecticesIndexes.push_back(Vector<int>(atoi(splittedTriangles[0].c_str())-1,atoi(splittedTriangles[1].c_str())-1,atoi(splittedTriangles[2].c_str())-1));
                    }
                    trianglesVecticesIndexes.push_back(vecticesIndexes);
                }    
            }
        };

        ~Obj() {};

        uint nbTriangles = 0;
        uint failedTriangles = 0;

        void addVertices(Vector<float> vertex) {
            v.push_back(vertex);
        }

        void addTextureVertices(Vector<float> vertex) {
            vt.push_back(vertex);
        }

        void addNormalVertices(Vector<float> vertex) {
            vn.push_back(vertex);
        }

        std::vector<Vector<float>> getVertices() {
            return v;
        }

        std::vector<Vector<float>> getTextureVertices() {
            return vt;
        }

        std::vector<Vector<float>> getNormalVertices() {
            return vn;
        }

        std::vector<std::vector<Vector<int>>> getIndexes() {
            return trianglesVecticesIndexes;
        }

        void print() {
            for (uint i=0;i<v.size();i++) {
                v[i].printCoord();
            }

            for (uint i=0;i<vt.size();i++) {
                vt[i].printCoord();
            }

            for (uint i=0;i<vn.size();i++) {
                vn[i].printCoord();
            }

            for (uint i=0;i<trianglesVecticesIndexes.size();i++) {
                std::cout << "f" << std::endl;
                for (uint j=0;j<trianglesVecticesIndexes[j].size();j++) {
                    trianglesVecticesIndexes[i][j].printCoord();
                }
            }
        }
};
