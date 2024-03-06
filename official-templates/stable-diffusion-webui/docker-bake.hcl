variable "RELEASE" {
    default = "11.0.0"
}

target "default" {
  dockerfile = "Dockerfile"
  tags = ["runpod/stable-diffusion:web-ui-${RELEASE}"]
  contexts = {
    scripts = "../../container-template"
    proxy = "../../container-template/proxy"
  }
  args = {
    WEBUI_VERSION = "v1.8.0"
    TORCH_VERSION = "2.0.1"
    XFORMERS_EVRSION = "0.0.22"
  }
}
