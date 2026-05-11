#ifndef BOARD_H
#define BOARD_H

#include <http.h>
#include <web_socket.h>
#include <mqtt.h>
#include <udp.h>
#include <network_interface.h>

#include <string>
#include <cstdint>     

#include "led/led.h"
#include "backlight.h"
#include "camera.h"

void* create_board();

class AudioCodec;
class Display;

class Board {
private:
    Board(const Board&) = delete;           
    Board& operator=(const Board&) = delete; 
protected:
    Board();
    std::string GenerateUuid();
    std::string uuid_;

public:
    static Board& GetInstance() {
        static Board* instance = static_cast<Board*>(create_board());
        return *instance;
    }

    virtual ~Board() = default;

    virtual std::string GetBoardType() = 0;
    virtual std::string GetUuid() { return uuid_; }
    virtual Backlight* GetBacklight() { return nullptr; }
    virtual Led* GetLed();
    virtual AudioCodec* GetAudioCodec() = 0;
    virtual bool GetTemperature(float& esp32temp);
    virtual Display* GetDisplay();
    virtual Camera* GetCamera();
    virtual NetworkInterface* GetNetwork() = 0;
    virtual void StartNetwork() = 0;
    virtual const char* GetNetworkStateIcon() = 0;
    virtual bool GetBatteryLevel(int &level, bool& charging, bool& discharging);
    virtual std::string GetJson();
    virtual void SetPowerSaveMode(bool enabled) = 0;
    virtual std::string GetBoardJson() = 0;
    virtual std::string GetDeviceStatusJson() = 0;
    virtual bool SetFanSpeed(uint8_t speed) { (void)speed; return false; }  
    virtual bool HasFan() { return false; }  
    
    virtual bool HasEnvironmentSensor() { return false; }
    virtual bool GetEnvironment(float& temperature_c, float& humidity_rh) {
        (void)temperature_c;
        (void)humidity_rh;
        return false;
    }
    virtual bool HasLightSensor() { return false; }
    virtual bool GetLightAndProximity(uint16_t& als, uint16_t& ps) {
        (void)als;
        (void)ps;
        return false;
    }

    virtual bool HasUltrasound() { return false; }
    virtual bool UltrasoundGetDistanceMm(uint16_t& distance_mm) {
        (void)distance_mm;
        return false;
    }
    virtual bool UltrasoundSetColor(uint8_t r1, uint8_t g1, uint8_t b1,
                                    uint8_t r2, uint8_t g2, uint8_t b2) {
        (void)r1; (void)g1; (void)b1;
        (void)r2; (void)g2; (void)b2;
        return false;
    }
    virtual bool HasImu() { return false; }
    virtual bool ImuGetRaw(float& acc_x, float& acc_y, float& acc_z,
                           float& gyro_x, float& gyro_y, float& gyro_z) {
        (void)acc_x; (void)acc_y; (void)acc_z;
        (void)gyro_x; (void)gyro_y; (void)gyro_z;
        return false;
    }
    virtual void MatrixShowDistance(float distance) {} 
    virtual bool HasMatrix() { return false; }
    virtual bool MatrixSetBrightness(uint8_t brightness) { (void)brightness; return false; }
    virtual bool MatrixDisplayStatic(const uint8_t* data, size_t len) { (void)data; (void)len; return false; }
    virtual bool MatrixDisplayScroll(const uint8_t* data, size_t len) { (void)data; (void)len; return false; }

    //SerialBusServo
    virtual bool HasSerialServo() { return false; }
    virtual bool SerialServoSetPosition(int id, uint16_t position, uint16_t time) {
        (void)id;
        (void)position;
        (void)time;
        return false;  
    }
    virtual bool SerialServoReadPosition(int id, int& position) {
        (void)id;
        (void)position;
        return false;
    }
    virtual bool SerialServoReadID(int& id_read) {
        (void)id_read;
        return false;
    }
    virtual bool SerialServoUnload(int id) {
        (void)id;
        return false; 
    }
    virtual bool SerialServoLoad(int id) {
        (void)id;
        return false; 
    }

};

#define DECLARE_BOARD(BOARD_CLASS_NAME) \
void* create_board() { \
    return new BOARD_CLASS_NAME(); \
}

#endif // BOARD_H
