function erode(img, se) {
    
}

function changeState(block) {
    if (block.style.backgroundColor == "black") {
        block.style.background = "WHITE";
        SE[block.dataset.x][block.dataset.y] = 0;
    } else {
        block.style.background = "BLACK";
        SE[block.dataset.x][block.dataset.y] = 1;
    }
}

//Source: https://stackoverflow.com/questions/35302149/how-do-you-convert-an-image-png-to-a-2d-array-binary-image
function GetBinary() {
    //Create a canvas
    var canvas = document.getElementById("blank");

    //Create an object ctx to draw the image in the canvas

    var img = document.getElementById('imgrgb');
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;

    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    //Get r,g,b,a of the whole image

    var map = ctx.getImageData(0, 0, img.width, img.height);
    var imdata = map.data;

    var currentInnerArray;
    var zeroesAndOnes = [];
    var r, g, b;
    for (var p = 0, len = imdata.length; p < len; p += 4) {
        r = imdata[p]
        g = imdata[p + 1];
        b = imdata[p + 2];
        //Ignore a channel        

        // Each line is the pixel width * 4, (rgba), start a newline
        if (p % canvas.width * 4 === 0) {
            currentInnerArray = [];
            zeroesAndOnes.push(currentInnerArray);
        }

        if ((r >= 100 && r <= 255) && (g >= 0 && g <= 229) && (b >= 0 && b <= 255)) {
            currentInnerArray.push(1);
            // black  = water
            imdata[p] = 0;
            imdata[p + 1] = 0;
            imdata[p + 2] = 0;

        } else {
            currentInnerArray.push(0);
            // white = land
            imdata[p] = 255;
            imdata[p + 1] = 255;
            imdata[p + 2] = 255;
        }
    }
    ctx.putImageData(map, 0, 0);
    return zeroesAndOnes;
}

var SE = new Array(3);
for (var i = 0; i < 3; i++) {
    SE[i] = new Array(3);
    for (var j = 0; j < 3; j++)
        SE[i][j] = 0;
}

for (i = 0; i < 3; i++) {
    var br = document.createElement("br");
    for (j = 0; j < 3; j++) {
        var div = document.createElement("div");
        div.className = "sdBlock";
        div.dataset.x = i;
        div.dataset.y = j;
        div.addEventListener("click", function () { changeState(this) });
        document.getElementById("SE").appendChild(div);
    }
    document.getElementById("SE").appendChild(br);
}

/*for (i = 0; i < 100; i++) {
    var br = document.createElement("br");
    for (j = 0; j < 100; j++) {
        var div = document.createElement("div");
        div.className = "imageBlock";
        div.id = "I" + i + "x" + j;
        var id = "I" + i + "x" + j;
        div.addEventListener("click", function () { changeState(this) });
        document.getElementById("I").appendChild(div);
    }
    document.getElementById("I").appendChild(br);
}*/
