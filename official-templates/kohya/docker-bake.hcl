variable "RELEASE" {
    default = "1.0.5"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_kohya:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}