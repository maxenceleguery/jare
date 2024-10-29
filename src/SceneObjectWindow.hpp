#pragma once

#include <gtkmm/button.h>
#include <gtkmm/window.h>
#include <gtkmm/entry.h>
#include <gtkmm/box.h>
#include <gtkmm/grid.h>

class SceneObjectWindow : public Gtk::Window {
    private:
        Gtk::Grid grid;
        Gtk::Entry text;
    public:
        SceneObjectWindow() {
            set_title("Scene Object");
            set_border_width(10);
            add(grid);

            text.show();
            text.set_text("");
            text.set_editable(true);

            grid.attach(text, 0, 0);
            grid.show();
            
        };
        ~SceneObjectWindow() {};
};

