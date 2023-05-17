#pragma once
#include "Vector.hpp"

#include <vector>
#include <cstring>
#include <sstream>
#include <fstream>

class Obj {
    private:
        std::string nameObj;
        std::vector<Vector<double>> v;
        std::vector<Vector<double>> vt;
        std::vector<Vector<double>> vn;
        std::vector<std::vector<Vector<int>>> facesVecticesIndexes;

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
            std::cout << "Loading " << name.c_str() << std::endl;
            std::string path = std::string("./models/");
            std::ifstream monFlux((path+name).c_str());
            std::string ligne;

            while (getline(monFlux, ligne)) {
                std::vector<std::string> splitted = split(ligne, ' ');
                if (splitted[0] == "o") {
                    nameObj = splitted[1];
                }
                else if (splitted[0] == "v") {
                    double x = atof(splitted[1].c_str());
                    double y = atof(splitted[2].c_str());
                    double z = atof(splitted[3].c_str());
                    v.push_back(Vector<double>(x, y, z));
                }
                else if (splitted[0] == "vt") {
                    double xt = atof(splitted[1].c_str());
                    double yt = atof(splitted[2].c_str());
                    vt.push_back(Vector<double>(xt, yt, 0.));
                }
                else if (splitted[0] == "vn") {
                    double xn = atof(splitted[1].c_str());
                    double yn = atof(splitted[2].c_str());
                    double zn = atof(splitted[3].c_str());
                    vn.push_back(Vector<double>(xn, yn, zn));
                }
                else if (splitted[0] == "f") {
                    splitted = split(rtrim(ligne), ' ');
                    std::vector<Vector<int>> vecticesIndexes;
                    for (uint i=1;i<splitted.size();i++) {
                        std::vector<std::string> splittedFaces = split(splitted[i], '/');
                        vecticesIndexes.push_back(Vector<int>(atoi(splittedFaces[0].c_str())-1,atoi(splittedFaces[1].c_str())-1,atoi(splittedFaces[2].c_str())-1));
                    }
                    facesVecticesIndexes.push_back(vecticesIndexes);
                }    
            }
        };

        ~Obj() {};

        void addVertices(Vector<double> vertex) {
            v.push_back(vertex);
        }

        void addTextureVertices(Vector<double> vertex) {
            vt.push_back(vertex);
        }

        void addNormalVertices(Vector<double> vertex) {
            vn.push_back(vertex);
        }

        std::vector<Vector<double>> getVertices() {
            return v;
        }

        std::vector<Vector<double>> getTextureVertices() {
            return vt;
        }

        std::vector<Vector<double>> getNormalVertices() {
            return vn;
        }

        std::vector<std::vector<Vector<int>>> getIndexes() {
            return facesVecticesIndexes;
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

            for (uint i=0;i<facesVecticesIndexes.size();i++) {
                std::cout << "f" << std::endl;
                for (uint j=0;j<facesVecticesIndexes[j].size();j++) {
                    facesVecticesIndexes[i][j].printCoord();
                }
            }
        }
};
