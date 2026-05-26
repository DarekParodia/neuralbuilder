#include "logger.h"
#include "neuron.h"
#include "app-window.h"

#include <vector>

int main() {
    neural::network netw;

    netw.load_json("../example_network.json");
    netw.print();

    auto ui = AppWindow::create();

    ui->on_request_increase_value([&] {
        ui->set_counter(ui->get_counter() + 1);
    });

    ui->run();
}