variable "RELEASE" {
    default = "1.0.1"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_animatediff:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
