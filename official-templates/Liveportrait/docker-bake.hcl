variable "RELEASE" {
    default = "1.0.7"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_liveportrait:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
