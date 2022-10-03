let audio = document.querySelector("#audio")

document.querySelector("#grabar").addEventListener("click", function(ev){
    navigator.mediaDevices.getUserMedia({audio: true, video:true})
        .then(record)
        .catch(err => console.log(err))
})

function record(stream){
    audio.srcObject = stream;
}