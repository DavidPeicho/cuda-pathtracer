#include <glad/glad.h>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw_gl3.h"
#include "gui_manager.h"

namespace gui
{
  namespace
  {
    bool vectorGetter(void* data, int n, const char** out_text)
    {
      const std::vector<std::string>* v = (std::vector<std::string>*)data;
      *out_text = (*v)[n].c_str();
      return true;
    }
  }

  GUIManager *GUIManager::_instance = nullptr;

  GUIManager::GUIManager()
  { }

  void
  GUIManager::init(GLFWwindow* window)
  {
    ImGui_ImplGlfwGL3_Init(window, true);
  }

  void
  GUIManager::begin()
  {
    ImGui_ImplGlfwGL3_NewFrame();
  }

  float
  GUIManager::info(int& scene_id, const std::vector<std::string>& items)
  {
    ImGui::Begin("Info", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("FPS: %.1f", 1000.0f / ImGui::GetIO().Framerate);
    float height = ImGui::GetWindowHeight();
    ImGui::ListBox("Loaded scenes", &scene_id, vectorGetter, (void*)&items, (int)items.size(), -1);
    ImGui::End();

    return height;
  }

  void
  GUIManager::camera(scene::Camera& cam, float h_offset)
  {
    constexpr float MAX_POS = 1000.0f;
    (void) h_offset;
    //ImGui::SetNextWindowPos(ImVec2(0, 100), ImGuiCond_Once);
    ImGui::Begin("Camera");
    ImGui::SliderFloat("Speed", &cam.speed, 1.0f, 10.0f);
    ImGui::SliderFloat("Fov X", &cam.fov_x, 1.2f, 2.0f);
    ImGui::Separator();
    ImGui::InputFloat("x", &cam.position.x, -MAX_POS, MAX_POS);
    ImGui::InputFloat("y", &cam.position.y, -MAX_POS, MAX_POS);
    ImGui::InputFloat("z", &cam.position.z, -MAX_POS, MAX_POS);
    ImGui::End();
  }

  void
  GUIManager::render()
  {
    ImGui::Render();
  }

  void
  GUIManager::release()
  {
    ImGui_ImplGlfwGL3_Shutdown();
  }

} // namespace gui
