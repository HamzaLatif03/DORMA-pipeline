// Presage SmartSpectra video processor.
// Reads a video file, runs spot-mode vitals extraction via the SDK,
// and prints JSON results to stdout.
//
// Usage: ./presage_processor --api_key=KEY --video=path.mp4

#include <smartspectra/container/foreground_container.hpp>
#include <smartspectra/container/settings.hpp>
#include <physiology/modules/messages/metrics.h>
#include <glog/logging.h>

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <mutex>

using namespace presage::smartspectra;

static std::mutex g_mutex;
static float g_pulse = -1.0f;
static float g_breathing = -1.0f;
static float g_hrv_sdnn = -1.0f;
static float g_hrv_rmssd = -1.0f;
static bool g_has_result = false;

static std::string parse_arg(int argc, char** argv, const std::string& prefix) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind(prefix, 0) == 0)
            return a.substr(prefix.size());
    }
    return "";
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    FLAGS_minloglevel = 2;

    std::string api_key = parse_arg(argc, argv, "--api_key=");
    std::string video_path = parse_arg(argc, argv, "--video=");

    if (api_key.empty()) {
        const char* env_key = std::getenv("PRESAGE_API_KEY");
        if (env_key) api_key = env_key;
    }

    if (api_key.empty() || video_path.empty()) {
        std::cerr << "Usage: presage_processor --api_key=KEY --video=path.mp4\n";
        std::cout << "{\"error\":\"missing arguments\"}\n";
        return 1;
    }

    try {
        container::settings::Settings<
            container::settings::OperationMode::Spot,
            container::settings::IntegrationMode::Rest
        > settings;

        settings.video_source.input_video_path = video_path;
        settings.video_source.device_index = -1;
        settings.headless = true;
        settings.enable_edge_metrics = false;
        settings.verbosity_level = 0;
        settings.integration.api_key = api_key;

        auto ctr = std::make_unique<
            container::CpuSpotRestForegroundContainer>(settings);

        auto status = ctr->SetOnCoreMetricsOutput(
            [](const presage::physiology::MetricsBuffer& metrics, int64_t) {
                std::lock_guard<std::mutex> lock(g_mutex);

                if (!metrics.pulse().rate().empty()) {
                    g_pulse = metrics.pulse().rate().rbegin()->value();
                }
                if (!metrics.pulse().strict().value() > 0) {
                    g_pulse = metrics.pulse().strict().value();
                }
                if (!metrics.breathing().rate().empty()) {
                    g_breathing = metrics.breathing().rate().rbegin()->value();
                }
                if (!metrics.breathing().strict().value() > 0) {
                    g_breathing = metrics.breathing().strict().value();
                }

                g_has_result = true;
                return absl::OkStatus();
            }
        );

        if (!status.ok()) {
            std::cout << "{\"error\":\"failed to set metrics callback: "
                      << status.message() << "\"}\n";
            return 1;
        }

        auto init_status = ctr->Initialize();
        if (!init_status.ok()) {
            std::cout << "{\"error\":\"init failed: "
                      << init_status.message() << "\"}\n";
            return 1;
        }

        auto run_status = ctr->Run();

        // Output JSON result
        std::lock_guard<std::mutex> lock(g_mutex);
        std::ostringstream json;
        json << "{";
        if (g_has_result) {
            json << "\"hr\":" << g_pulse
                 << ",\"rr\":" << g_breathing;
            if (g_hrv_sdnn >= 0)
                json << ",\"hrv_sdnn\":" << g_hrv_sdnn;
            if (g_hrv_rmssd >= 0)
                json << ",\"hrv_rmssd\":" << g_hrv_rmssd;
        } else {
            json << "\"error\":\"no vitals detected\"";
        }
        json << "}";
        std::cout << json.str() << "\n";
        return g_has_result ? 0 : 1;

    } catch (const std::exception& e) {
        std::cout << "{\"error\":\"" << e.what() << "\"}\n";
        return 1;
    }
}
