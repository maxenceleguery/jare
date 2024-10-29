#pragma once

#include "SceneObject.hpp"

#include <gtkmm/button.h>
#include <gtkmm/window.h>
#include <gtkmm/entry.h>
#include <gtkmm/box.h>
#include <gtkmm/grid.h>

#include <vector>

class SceneObjectWindow : public Gtk::Window {
    private:
        Gtk::Grid grid;

        std::vector<Gtk::Entry> offset;
        std::vector<Gtk::Label> offset_text;
        std::vector<Gtk::Entry> scale;
        std::vector<Gtk::Label> scale_text;
        std::vector<Gtk::Entry> rotation;
        std::vector<Gtk::Label> rotation_text;

        Gtk::Label offset_label;
        Gtk::Label scale_label;
        Gtk::Label rotation_label;

        void setupVectorDisplay(const std::string& name, const Vector<float> value, Gtk::Label& title, std::vector<Gtk::Entry>& entries, std::vector<Gtk::Label>& labels, const int line) {
            std::string axis = "XYZ";

            entries = std::vector<Gtk::Entry>(3);
            labels = std::vector<Gtk::Label>(3);
            title.show();
            title.set_text(name);
            grid.attach(title, 0, line*4);
            for (int i = 0; i<3; i++) {
                labels[i].show();
                labels[i].set_text(Glib::ustring(axis.substr(i, 1)));
                grid.attach(labels[i], 0, line*4+i+1);
                entries[i].show();
                entries[i].set_text(Glib::ustring(std::to_string(value[i])));
                offset[i].set_editable(true);
                grid.attach(entries[i], 2, line*4+i+1);
            }
        }
    public:
        SceneObjectWindow(SceneObject* obj) {
            set_title("Scene Object");
            set_border_width(10);
            add(grid);

            setupVectorDisplay("Offset", obj->transforms[0], offset_label, offset, offset_text, 0);
            setupVectorDisplay("Scale", obj->transforms[1], scale_label, scale, scale_text, 1);
            setupVectorDisplay("Rotation", obj->transforms[2], rotation_label, rotation, rotation_text, 2);

            grid.show();
        };
        ~SceneObjectWindow() {};
};

