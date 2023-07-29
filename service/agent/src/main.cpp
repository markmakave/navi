#include <esp_log.h>
#include <esp_wifi.h>
#include <esp_camera.h>

#define ESP_CALL(x) { if (x != ESP_OK) ESP_LOGE("CALL", "Error occured"); }
extern "C" {

void app_main() {

    wifi_init_config_t wifi_config{};
    
    ESP_CALL(esp_wifi_init(&wifi_config));

    while(true) {}

}

}
