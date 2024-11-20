variable "RELEASE" {
    default = "1.0.2"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_comfy:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
