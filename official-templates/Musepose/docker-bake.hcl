variable "RELEASE" {
    default = "1.0.3"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_musepose:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
