variable "RELEASE" {
    default = "1.0.6"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_mimicmotion:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
