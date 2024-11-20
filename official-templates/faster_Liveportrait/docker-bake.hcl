variable "RELEASE" {
    default = "1.1.5"
}

target "default" {
    dockerfile = "Dockerfile2"
    tags = ["wlsdml1114/engui_faster_liveportrait:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
