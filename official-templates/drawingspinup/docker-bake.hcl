variable "RELEASE" {
    default = "1.0.0"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_drawingspinup:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
