/*
 * MCP Server Implementation
 * Reference: https://modelcontextprotocol.io/specification/2024-11-05
 */

#include "mcp_server.h"
#include <esp_log.h>
#include <esp_app_desc.h>
#include <algorithm>
#include <cstring>
#include <esp_pthread.h>

#include "application.h"
#include "display.h"
#include "board.h"
#include <cmath>
#define TAG "MCP"

#define DEFAULT_TOOLCALL_STACK_SIZE 6144

McpServer::McpServer() {
}

McpServer::~McpServer() {
    for (auto tool : tools_) {
        delete tool;
    }
    tools_.clear();
}

void McpServer::AddCommonTools() {
    auto original_tools = std::move(tools_);
    auto& board = Board::GetInstance();
    auto camera = board.GetCamera();
    // ========= 超声波：读距离 + RGB 灯独立控制 =========
    if (board.HasEnvironmentSensor()) {
        AddTool("self.env.get_temperature_humidity",
            "读取当前环境温度和湿度，单位分别为摄氏度和相对湿度百分比。",
            PropertyList(),
            [&board](const PropertyList& properties) -> ReturnValue {
                (void)properties;
                float t = 0.0f;
                float h = 0.0f;
                if (!board.GetEnvironment(t, h)) {
                    return std::string("{\"success\":false,\"message\":\"read environment failed\"}");
                }
                char buf[128];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,\"temperature_c\":%.2f,\"humidity_rh\":%.2f}",
                         t, h);
                return std::string(buf);
            });
    }
    if (board.HasEnvironmentSensor() && board.HasFan()) {
        AddTool("self.env.auto_fan_control",
            "根据当前温度和湿度自动调节风扇转速。\n"
            "大致策略：\n"
            "  温度或湿度较高时风扇高速；\n"
            "  中等时中速；\n"
            "  较低时关闭风扇。",
            PropertyList(),
            [&board](const PropertyList& properties) -> ReturnValue {
                (void)properties;
                float t = 0.0f;
                float h = 0.0f;
                if (!board.GetEnvironment(t, h)) {
                    return std::string("{\"success\":false,\"message\":\"read environment failed\"}");
                }

                uint8_t speed = 0;
                if (t >= 30.0f || h >= 70.0f) {
                    speed = 200;
                } else if (t >= 26.0f || h >= 60.0f) {
                    speed = 180;
                } else if (t >= 24.0f || h >= 50.0f) {
                    speed = 120;
                } else {
                    speed = 0;
                }

                if (!board.SetFanSpeed(speed)) {
                    return std::string("{\"success\":false,\"message\":\"SetFanSpeed failed\"}");
                }

                char buf[160];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,"
                         "\"temperature_c\":%.2f,"
                         "\"humidity_rh\":%.2f,"
                         "\"fan_speed\":%u}",
                         t, h, (unsigned)speed);
                return std::string(buf);
            });
    }
    if (board.HasLightSensor()) {
        AddTool("self.env.get_light_ps",
            "读取环境光强 ALS 和接近传感 PS 的原始数值。",
            PropertyList(),
            [&board](const PropertyList& properties) -> ReturnValue {
                (void)properties;
                uint16_t als = 0;
                uint16_t ps = 0;
                if (!board.GetLightAndProximity(als, ps)) {
                    return std::string("{\"success\":false,\"message\":\"read light/ps failed\"}");
                }
                char buf[160];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,"
                         "\"als\":%u,"
                         "\"ps\":%u}",
                         (unsigned)als, (unsigned)ps);
                return std::string(buf);
            });
    }
    
    if (board.HasEnvironmentSensor() && board.HasMatrix()) {
        AddTool("self.demo.temp_emoji_and_say",
            "读取温度湿度，并在点阵显示表情（冷/正常/热），同时返回建议播报文本给小智。",
            PropertyList({
                Property("cold_c", kPropertyTypeInteger, 0),
                Property("hot_c",  kPropertyTypeInteger, 0),
            }),
            [&board](const PropertyList& props) -> ReturnValue {
                int cold_c = props["cold_c"].value<int>();
                int hot_c  = props["hot_c"].value<int>();
                if (cold_c == 0) cold_c = 18;
                if (hot_c  == 0) hot_c  = 30;

                float t = 0.0f, h = 0.0f;
                if (!board.GetEnvironment(t, h)) {
                    return std::string("{\"success\":false,\"message\":\"GetEnvironment failed\"}");
                }

                static const uint8_t EMOJI_OK_LOCAL[16] = {
                0x3C,0x42,0xA5,0x81,0xA5,0x99,0x42,0x3C,
                0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
                };
                static const uint8_t EMOJI_COLD_LOCAL[16] = {
                0x3C,0x42,0xA5,0x81,0xBD,0x81,0x42,0x3C,
                0x18,0x24,0x42,0x81,0x42,0x24,0x18,0x00
                };
                static const uint8_t EMOJI_HOT_LOCAL[16] = {
                0x3C,0x42,0xA5,0x81,0x99,0xA5,0x42,0x3C,
                0x10,0x38,0x7C,0xFE,0x7C,0x38,0x10,0x00
                };

                const uint8_t* emoji = EMOJI_OK_LOCAL;
                const char* state = "舒适";
                if (t <= (float)cold_c) { emoji = EMOJI_COLD_LOCAL; state = "偏冷"; }
                else if (t >= (float)hot_c) { emoji = EMOJI_HOT_LOCAL; state = "偏热"; }

                if (!board.MatrixDisplayStatic(emoji, 16)) {
                    return std::string("{\"success\":false,\"message\":\"MatrixDisplayStatic failed\"}");
                }

                char say[160];
                snprintf(say, sizeof(say), "当前温度%.1f度，湿度%.1f%%，体感%s。", t, h, state);

                char buf[260];
                snprintf(buf, sizeof(buf),
                        "{\"success\":true,\"temperature_c\":%.2f,\"humidity_rh\":%.2f,\"state\":\"%s\",\"say\":\"%s\"}",
                        t, h, state, say);
                return std::string(buf);
            });
    }


    if (board.HasLightSensor() && board.HasUltrasound()) {
        AddTool("self.env.light_auto_ultrasound",
            "根据环境光强自动控制超声波模块上的绿灯。\n"
            "如果太暗就点亮红灯，如果够亮则关闭绿灯。",
            PropertyList(),
            [&board](const PropertyList& properties) -> ReturnValue {
                (void)properties;
                uint16_t als = 0;
                uint16_t ps = 0;
                if (!board.GetLightAndProximity(als, ps)) {
                    return std::string("{\"success\":false,\"message\":\"read light/ps failed\"}");
                }

                bool dark = als < 700;
                bool ok;
                if (dark) {
                    ok = board.UltrasoundSetColor(0, 255, 0, 0, 255, 0);
                } else {
                    ok = board.UltrasoundSetColor(0, 0, 0, 0, 0, 0);
                }
                if (!ok) {
                    return std::string("{\"success\":false,\"message\":\"UltrasoundSetColor failed\"}");
                }

                char buf[200];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,"
                         "\"als\":%u,"
                         "\"ps\":%u,"
                         "\"dark\":%s}",
                         (unsigned)als,
                         (unsigned)ps,
                         dark ? "true" : "false");
                return std::string(buf);
            });
    }
    
    if (board.HasImu()) {
        AddTool("self.imu.get_raw",
            "读取 IMU 的六轴原始数据，包含加速度 (g) 和角速度 (dps)。",
            PropertyList(),
            [&board](const PropertyList& properties) -> ReturnValue {
                (void)properties;
                float ax = 0, ay = 0, az = 0;
                float gx = 0, gy = 0, gz = 0;
                if (!board.ImuGetRaw(ax, ay, az, gx, gy, gz)) {
                    return std::string("{\"success\":false,\"message\":\"ImuGetRaw failed\"}");
                }
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,"
                         "\"acc_x_g\":%.4f,"
                         "\"acc_y_g\":%.4f,"
                         "\"acc_z_g\":%.4f,"
                         "\"gyro_x_dps\":%.4f,"
                         "\"gyro_y_dps\":%.4f,"
                         "\"gyro_z_dps\":%.4f}",
                         ax, ay, az, gx, gy, gz);
                return std::string(buf);
            });
    }
    if (board.HasImu()) {
        AddTool("self.imu.get_attitude",
            "根据当前 IMU 数据估算姿态角，包括 pitch(俯仰) 和 roll(横滚)，单位为度。",
            PropertyList(),
            [&board](const PropertyList& properties) -> ReturnValue {
                (void)properties;
                float ax = 0, ay = 0, az = 0;
                float gx = 0, gy = 0, gz = 0;
                if (!board.ImuGetRaw(ax, ay, az, gx, gy, gz)) {
                    return std::string("{\"success\":false,\"message\":\"ImuGetRaw failed\"}");
                }

                float roll_rad  = atan2f(ay, az);
                float pitch_rad = atan2f(-ax, sqrtf(ay * ay + az * az));
                float roll_deg  = roll_rad  * 180.0f / (float)M_PI;
                float pitch_deg = pitch_rad * 180.0f / (float)M_PI;

                char buf[320];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,"
                         "\"pitch_deg\":%.2f,"
                         "\"roll_deg\":%.2f,"
                         "\"acc_x_g\":%.4f,"
                         "\"acc_y_g\":%.4f,"
                         "\"acc_z_g\":%.4f,"
                         "\"gyro_x_dps\":%.4f,"
                         "\"gyro_y_dps\":%.4f,"
                         "\"gyro_z_dps\":%.4f}",
                         pitch_deg, roll_deg,
                         ax, ay, az, gx, gy, gz);
                return std::string(buf);
            });
    }

    if (board.HasUltrasound()) {
        AddTool("self.ultrasound.get_distance",
            "读取前方超声波距离，单位 mm。\n"
            "你可以用它来判断前面是否有东西、离得远不远，\n"
            "再结合RGB灯来做简单的避障逻辑。\n",
            PropertyList(),   
            [&board](const PropertyList& properties) -> ReturnValue {
                uint16_t dist_mm = 0;
                                if (!board.UltrasoundGetDistanceMm(dist_mm)) {
                    return std::string("{\"success\":false, \"message\":\"read failed\"}");
                }

                board.MatrixShowDistance((float)dist_mm);

                printf("Ultrasound Read: %d mm\n", dist_mm);

                char buf[64];
                snprintf(buf, sizeof(buf), "{\"success\":true,\"distance_mm\":%u}", dist_mm);
                return std::string(buf);
            });
            
        AddTool("self.fan.set_speed",
            "设置风扇转速（直接发送 I2C，0 关风扇，数值越大风越大）。\n"
            "参数：\n"
            "  speed：整数，建议先用 0 和 60 测一下效果。\n",
            PropertyList({
                Property("speed", kPropertyTypeInteger, -255, 255),
            }),
            [&board](const PropertyList& properties) -> ReturnValue {
                int speed = properties["speed"].value<int>();
                if (speed < 0)   speed = -255;
                if (speed > 255) speed = 255;

                if (!board.SetFanSpeed(static_cast<uint8_t>(speed))) {
                    return std::string("{\"success\":false,\"message\":\"SetFanSpeed failed\"}");
                }
                return std::string("{\"success\":true}");
            });
        
        // 独立设置超声波模块 RGB 灯颜色
        AddTool("self.ultrasound.set_rgb",
            "单独设置超声波模块上的 RGB 灯颜色。\n"
            "参数：\n"
            "  led_r / led_g / led_b：颜色分量，范围 0~255。\n"
            "示例：\n"
            "  - 设为蓝色：led_r=0, led_g=0, led_b=255；\n"
            "  - 关灯：led_r=0, led_g=0, led_b=0。\n"
            "左右两颗灯会被设置为同一种颜色。\n",
            PropertyList({
                Property("led_r", kPropertyTypeInteger, 0, 255),
                Property("led_g", kPropertyTypeInteger, 0, 255),
                Property("led_b", kPropertyTypeInteger, 0, 255),
            }),
            [&board](const PropertyList& properties) -> ReturnValue {
                int led_r = properties["led_r"].value<int>();
                int led_g = properties["led_g"].value<int>();
                int led_b = properties["led_b"].value<int>();

                bool ok = board.UltrasoundSetColor(
                    (uint8_t)led_r, (uint8_t)led_g, (uint8_t)led_b,
                    (uint8_t)led_r, (uint8_t)led_g, (uint8_t)led_b
                );
                if (!ok) {
                    return std::string("{\"success\":false,"
                                       "\"message\":\"UltrasoundSetColor failed\"}");
                }
                return std::string("{\"success\":true}");
            });
    }

    //总线舵机控制工具
    if(board.HasSerialServo()){
        AddTool("self.serial_servo.read_id",
            "读取总线舵机的ID,当ID为-2048时表示读取失败\n",
            PropertyList(),   
            [this,&board](const PropertyList& properties) -> ReturnValue {
                if (!board.SerialServoReadID(SerialServoID)) {
                    return std::string("{\"success\":false,"
                                       "\"message\":\"read servo id failed\"}");
                }
                char buf[64];
                snprintf(buf, sizeof(buf),
                         "{\"success\":true,\"servo id\":%u}", SerialServoID);
                return std::string(buf);
            });
        
            AddTool("self.serial_servo.read_position",
                "读取总线舵机当前的位置，当位置为-2048时表示读取失败\n"
                "若是没有指定舵机ID，那么就调用self.serial_servo.read_id工具先读取ID，然后再调用此工具\n",
                PropertyList(),   
                [this,&board](const PropertyList& properties) -> ReturnValue {
                    int servo_postion = -2048;
                    if (!board.SerialServoReadPosition(SerialServoID,servo_postion)) {
                        return std::string("{\"success\":false,"
                                           "\"message\":\"read servo postion failed\"}");
                    }
                    char buf[64];
                    snprintf(buf, sizeof(buf),
                             "{\"success\":true,\"read servo postion\":%d}", servo_postion);
                    return std::string(buf);
                });

            AddTool("self.serial_servo.set_position",
                "舵机位置控制指令\n"
                "参数：\n"
                "servo_id:需要控制的总线舵机id\n"
                "servo_postion:需要设置的舵机目标位置，范围是0~1000，中位是500\n"
                "time_ms:舵机移动到目标位置的时间，单位毫秒,默认500ms\n"
                "说明：\n"
                "若是没有指定舵机ID，那么就调用self.serial_servo.read_id工具先读取ID，然后再调用此工具\n",
                PropertyList({
                    Property("servo_id",          kPropertyTypeInteger, 0),
                    Property("servo_postion",            kPropertyTypeInteger, 0),
                    Property("time_ms",           kPropertyTypeInteger, 500),
                }),  
                [this,&board](const PropertyList& properties) -> ReturnValue {
                    int servo_id = properties["servo_id"].value<int>();
                    int servo_postion = properties["servo_postion"].value<int>();
                    int time_ms = properties["time_ms"].value<int>();

                    if (!board.SerialServoSetPosition(servo_id,servo_postion,time_ms)) {
                        return std::string("{\"success\":false,"
                                            "\"message\":\"set servo postion failed\"}");
                    }
                    char buf[64];
                    snprintf(buf, sizeof(buf),
                                "{\"success\":true,\"set servo postion\":%u}", SerialServoID);
                    SerialServoState = true;
                    return std::string(buf);
                });

                AddTool("self.serial_servo.load_controller",
                    "舵机上电、掉电控制指令\n"
                    "参数：\n"
                    "servo_load:需要控制的舵机状态，为true是上电，为false是掉电\n"
                    "若是没有指定舵机ID，那么就调用self.serial_servo.read_id工具先读取ID，然后再调用此工具\n",
                    PropertyList({
                        Property("servo_load",          kPropertyTypeBoolean, true),
                    }),  
                    [this,&board](const PropertyList& properties) -> ReturnValue {
                        SerialServoState = properties["servo_load"].value<bool>();
                        
                        if(SerialServoState == true){
                            if (!board.SerialServoLoad(SerialServoID)) {
                                return std::string("{\"success\":false,"
                                                    "\"message\":\"set servo load failed\"}");
                            }
                            char buf[64];
                            snprintf(buf, sizeof(buf),
                                        "{\"success\":true,\"set servo load\"}");
                            return std::string(buf);
                        }else{
                            if (!board.SerialServoUnload(SerialServoID)) {
                                return std::string("{\"success\":false,"
                                                    "\"message\":\"set servo unload failed\"}");
                            }
                            char buf[64];
                            snprintf(buf, sizeof(buf),
                                        "{\"success\":true,\"set servo unload\"}");
                            return std::string(buf);
                        }

                    });
    }

    // ========= 设备状态 / 音量 / 屏幕 =========
    AddTool("self.get_device_status",
        "获取设备的实时状态，包括音量、屏幕、网络、电池等信息。\n"
        "用途：\n"
        "1. 回答关于当前设备状态的问题。\n"
        "2. 在调整音量、亮度等之前，先了解当前状态。\n",
        PropertyList(),
        [&board](const PropertyList& properties) -> ReturnValue {
            return board.GetDeviceStatusJson();
        });

    AddTool("self.audio_speaker.set_volume", 
        "设置扬声器音量（0~100）。\n"
        "如果你不知道当前音量，可以先调用 self.get_device_status 再决定设置多少。\n",
        PropertyList({
            Property("volume", kPropertyTypeInteger, 0, 100)
        }), 
        [&board](const PropertyList& properties) -> ReturnValue {
            auto codec = board.GetAudioCodec();
            codec->SetOutputVolume(properties["volume"].value<int>());
            return true;
        });
    
    auto backlight = board.GetBacklight();
    if (backlight) {
        AddTool("self.screen.set_brightness",
            "设置屏幕亮度，范围 0~100。\n",
            PropertyList({
                Property("brightness", kPropertyTypeInteger, 0, 100)
            }),
            [backlight](const PropertyList& properties) -> ReturnValue {
                uint8_t brightness = static_cast<uint8_t>(properties["brightness"].value<int>());
                backlight->SetBrightness(brightness, true);
                return true;
            });
    }

    auto display = board.GetDisplay();
    if (display && !display->GetTheme().empty()) {
        AddTool("self.screen.set_theme",
            "设置屏幕主题，可选值：\"light\" 或 \"dark\"。\n",
            PropertyList({
                Property("theme", kPropertyTypeString)
            }),
            [display](const PropertyList& properties) -> ReturnValue {
                display->SetTheme(properties["theme"].value<std::string>().c_str());
                return true;
            });
    }

    if (camera) {
        AddTool("self.camera.take_photo",
            "拍一张照片并用文字解释当前画面。\n"
            "调用后你会得到一段对画面的解释文字，\n"
            "你可以把这段文字读给用户听，告诉他你看到了什么。\n"
            "参数：\n"
            "  question：你想就当前画面提问的问题，例如“画面里最显眼的东西是什么？”。\n",
            PropertyList({
                Property("question", kPropertyTypeString)
            }),
            [camera](const PropertyList& properties) -> ReturnValue {
                if (!camera->Capture()) {
                    return "{\"success\": false, \"message\": \"Failed to capture photo\"}";
                }
                auto question = properties["question"].value<std::string>();
                return camera->Explain(question);
            });
    }

    tools_.insert(tools_.end(), original_tools.begin(), original_tools.end());
}

void McpServer::AddTool(McpTool* tool) {
    // Prevent adding duplicate tools
    if (std::find_if(tools_.begin(), tools_.end(), [tool](const McpTool* t) { return t->name() == tool->name(); }) != tools_.end()) {
        ESP_LOGW(TAG, "Tool %s already added", tool->name().c_str());
        return;
    }

    ESP_LOGI(TAG, "Add tool: %s", tool->name().c_str());
    tools_.push_back(tool);
}

void McpServer::AddTool(const std::string& name, const std::string& description, const PropertyList& properties, std::function<ReturnValue(const PropertyList&)> callback) {
    AddTool(new McpTool(name, description, properties, callback));
}

void McpServer::ParseMessage(const std::string& message) {
    cJSON* json = cJSON_Parse(message.c_str());
    if (json == nullptr) {
        ESP_LOGE(TAG, "Failed to parse MCP message: %s", message.c_str());
        return;
    }
    ParseMessage(json);
    cJSON_Delete(json);
}

void McpServer::ParseCapabilities(const cJSON* capabilities) {
    auto vision = cJSON_GetObjectItem(capabilities, "vision");
    if (cJSON_IsObject(vision)) {
        auto url = cJSON_GetObjectItem(vision, "url");
        auto token = cJSON_GetObjectItem(vision, "token");
        if (cJSON_IsString(url)) {
            auto camera = Board::GetInstance().GetCamera();
            if (camera) {
                std::string url_str = std::string(url->valuestring);
                std::string token_str;
                if (cJSON_IsString(token)) {
                    token_str = std::string(token->valuestring);
                }
                camera->SetExplainUrl(url_str, token_str);
            }
        }
    }
}
void McpServer::ParseMessage(const cJSON* json) {
    // Check JSONRPC version
    auto version = cJSON_GetObjectItem(json, "jsonrpc");
    if (version == nullptr || !cJSON_IsString(version) || strcmp(version->valuestring, "2.0") != 0) {
        ESP_LOGE(TAG, "Invalid JSONRPC version: %s", version ? version->valuestring : "null");
        return;
    }
    
    // Check method
    auto method = cJSON_GetObjectItem(json, "method");
    if (method == nullptr || !cJSON_IsString(method)) {
        ESP_LOGE(TAG, "Missing method");
        return;
    }
    
    auto method_str = std::string(method->valuestring);
    if (method_str.find("notifications") == 0) {  
        return;
    }
    
    // Check params
    auto params = cJSON_GetObjectItem(json, "params");
    if (params != nullptr && !cJSON_IsObject(params)) {
        ESP_LOGE(TAG, "Invalid params for method: %s", method_str.c_str());
        return;
    }

    auto id = cJSON_GetObjectItem(json, "id");
    if (id == nullptr || !cJSON_IsNumber(id)) {
        ESP_LOGE(TAG, "Invalid id for method: %s", method_str.c_str());
        return;
    }
    auto id_int = id->valueint;
    
    if (method_str == "initialize") {
        if (cJSON_IsObject(params)) {
            auto capabilities = cJSON_GetObjectItem(params, "capabilities");
            if (cJSON_IsObject(capabilities)) {
                ParseCapabilities(capabilities);
            }
        }
        auto app_desc = esp_app_get_description();
        std::string message = "{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"" BOARD_NAME "\",\"version\":\"";
        message += app_desc->version;
        message += "\"}}";
        ReplyResult(id_int, message);
    } else if (method_str == "tools/list") {
        std::string cursor_str = "";
        if (params != nullptr) {
            auto cursor = cJSON_GetObjectItem(params, "cursor");
            if (cJSON_IsString(cursor)) {
                cursor_str = std::string(cursor->valuestring);
            }
        }
        GetToolsList(id_int, cursor_str);
    } else if (method_str == "tools/call") {
        if (!cJSON_IsObject(params)) {
            ESP_LOGE(TAG, "tools/call: Missing params");
            ReplyError(id_int, "Missing params");
            return;
        }
        auto tool_name = cJSON_GetObjectItem(params, "name");
        if (!cJSON_IsString(tool_name)) {
            ESP_LOGE(TAG, "tools/call: Missing name");
            ReplyError(id_int, "Missing name");
            return;
        }
        auto tool_arguments = cJSON_GetObjectItem(params, "arguments");
        if (tool_arguments != nullptr && !cJSON_IsObject(tool_arguments)) {
            ESP_LOGE(TAG, "tools/call: Invalid arguments");
            ReplyError(id_int, "Invalid arguments");
            return;
        }
        auto stack_size = cJSON_GetObjectItem(params, "stackSize");
        if (stack_size != nullptr && !cJSON_IsNumber(stack_size)) {
            ESP_LOGE(TAG, "tools/call: Invalid stackSize");
            ReplyError(id_int, "Invalid stackSize");
            return;
        }
        DoToolCall(id_int, std::string(tool_name->valuestring), tool_arguments, stack_size ? stack_size->valueint : DEFAULT_TOOLCALL_STACK_SIZE);
    } else {
        ESP_LOGE(TAG, "Method not implemented: %s", method_str.c_str());
        ReplyError(id_int, "Method not implemented: " + method_str);
    }
}

void McpServer::ReplyResult(int id, const std::string& result) {
    std::string payload = "{\"jsonrpc\":\"2.0\",\"id\":";
    payload += std::to_string(id) + ",\"result\":";
    payload += result;
    payload += "}";
    Application::GetInstance().SendMcpMessage(payload);
}

void McpServer::ReplyError(int id, const std::string& message) {
    std::string payload = "{\"jsonrpc\":\"2.0\",\"id\":";
    payload += std::to_string(id);
    payload += ",\"error\":{\"message\":\"";
    payload += message;
    payload += "\"}}";
    Application::GetInstance().SendMcpMessage(payload);
}

void McpServer::GetToolsList(int id, const std::string& cursor) {
    const int max_payload_size = 8000;
    std::string json = "{\"tools\":[";
    
    bool found_cursor = cursor.empty();
    auto it = tools_.begin();
    std::string next_cursor = "";
    
    while (it != tools_.end()) {
        // 如果我们还没有找到起始位置，继续搜索
        if (!found_cursor) {
            if ((*it)->name() == cursor) {
                found_cursor = true;
            } else {
                ++it;
                continue;
            }
        }
        
        std::string tool_json = (*it)->to_json() + ",";
        if (json.length() + tool_json.length() + 30 > max_payload_size) {
            next_cursor = (*it)->name();
            break;
        }
        
        json += tool_json;
        ++it;
    }
    
    if (json.back() == ',') {
        json.pop_back();
    }
    
    if (json.back() == '[' && !tools_.empty()) {
        // 如果没有添加任何tool，返回错误
        ESP_LOGE(TAG, "tools/list: Failed to add tool %s because of payload size limit", next_cursor.c_str());
        ReplyError(id, "Failed to add tool " + next_cursor + " because of payload size limit");
        return;
    }

    if (next_cursor.empty()) {
        json += "]}";
    } else {
        json += "],\"nextCursor\":\"" + next_cursor + "\"}";
    }
    
    ReplyResult(id, json);
}

void McpServer::DoToolCall(int id, const std::string& tool_name, const cJSON* tool_arguments, int stack_size) {
    auto tool_iter = std::find_if(tools_.begin(), tools_.end(), 
                                 [&tool_name](const McpTool* tool) { 
                                     return tool->name() == tool_name; 
                                 });
    
    if (tool_iter == tools_.end()) {
        ESP_LOGE(TAG, "tools/call: Unknown tool: %s", tool_name.c_str());
        ReplyError(id, "Unknown tool: " + tool_name);
        return;
    }

    PropertyList arguments = (*tool_iter)->properties();
    try {
        for (auto& argument : arguments) {
            bool found = false;
            if (cJSON_IsObject(tool_arguments)) {
                auto value = cJSON_GetObjectItem(tool_arguments, argument.name().c_str());
                if (argument.type() == kPropertyTypeBoolean && cJSON_IsBool(value)) {
                    argument.set_value<bool>(value->valueint == 1);
                    found = true;
                } else if (argument.type() == kPropertyTypeInteger && cJSON_IsNumber(value)) {
                    argument.set_value<int>(value->valueint);
                    found = true;
                } else if (argument.type() == kPropertyTypeString && cJSON_IsString(value)) {
                    argument.set_value<std::string>(value->valuestring);
                    found = true;
                }
            }

            if (!argument.has_default_value() && !found) {
                ESP_LOGE(TAG, "tools/call: Missing valid argument: %s", argument.name().c_str());
                ReplyError(id, "Missing valid argument: " + argument.name());
                return;
            }
        }
    } catch (const std::exception& e) {
        ESP_LOGE(TAG, "tools/call: %s", e.what());
        ReplyError(id, e.what());
        return;
    }

    // Start a task to receive data with stack size
    esp_pthread_cfg_t cfg = esp_pthread_get_default_config();
    cfg.thread_name = "tool_call";
    cfg.stack_size = stack_size;
    cfg.prio = 1;
    esp_pthread_set_cfg(&cfg);

    // Use a thread to call the tool to avoid blocking the main thread
    tool_call_thread_ = std::thread([this, id, tool_iter, arguments = std::move(arguments)]() {
        try {
            ReplyResult(id, (*tool_iter)->Call(arguments));
        } catch (const std::exception& e) {
            ESP_LOGE(TAG, "tools/call: %s", e.what());
            ReplyError(id, e.what());
        }
    });
    tool_call_thread_.detach();
}