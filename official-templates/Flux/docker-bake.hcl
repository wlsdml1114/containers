variable "RELEASE" {
    default = "1.0.8"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["wlsdml1114/engui_flux:${RELEASE}"]
    contexts = {
        scripts = "../../container-template"
        proxy = "../../container-template/proxy"
    }
}
