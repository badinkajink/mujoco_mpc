// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/log.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <mujoco/mujoco.h>

#include "mjpc/app.h"
#include "mjpc/tasks/tasks.h"

// gRPC — conditionally compiled so mjpc still builds without GRPC
#ifdef MJPC_BUILD_GRPC_SERVICE
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include "mjpc/grpc/ui_agent_service.h"
#endif

ABSL_FLAG(std::string, task, "Lean H12",
          "Which model to load on startup.");
ABSL_FLAG(int32_t, mjpc_port, 10000,
          "gRPC agent server port. 0 = disable gRPC.");

// machinery for replacing command line error by a macOS dialog box
// when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default")))
extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

int main(int argc, char** argv) {
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif
  absl::ParseCommandLine(argc, argv);

  std::string task_name = absl::GetFlag(FLAGS_task);
  auto tasks = mjpc::GetTasks();
  int task_id = -1;
  for (int i = 0; i < (int)tasks.size(); i++) {
    if (absl::EqualsIgnoreCase(task_name, tasks[i]->Name())) {
      task_id = i;
      break;
    }
  }
  if (task_id == -1) {
    std::cerr << "Invalid --task flag: '" << task_name
              << "'. Valid values:\n";
    for (int i = 0; i < (int)tasks.size(); i++) {
      std::cerr << "  " << tasks[i]->Name() << "\n";
    }
    mju_error("Invalid --task flag.");
  }

#ifdef MJPC_BUILD_GRPC_SERVICE
  int port = absl::GetFlag(FLAGS_mjpc_port);
  if (port > 0) {
    // Build app first so the Sim() pointer is valid before server starts
    mjpc::MjpcApp app(std::move(tasks), task_id);
    mjpc::agent_grpc::UiAgentService service(app.Sim());

    std::string server_address = absl::StrCat("[::]:", port);
    auto creds = grpc::experimental::LocalServerCredentials(LOCAL_TCP);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, creds);
    builder.SetMaxReceiveMessageSize(50 * 1024 * 1024);
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    LOG(INFO) << "gRPC agent server listening on " << server_address;

    app.Start();  // blocks until window closed
    server->Shutdown(absl::ToChronoTime(absl::Now() + absl::Seconds(1)));
    server->Wait();
  } else {
    mjpc::StartApp(tasks, task_id);
  }
#else
  mjpc::StartApp(tasks, task_id);
#endif

  return 0;
}
