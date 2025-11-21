const RADIUS = 5;
const VERSION = 12.1;
const NAME = 'Graphisoft';
const INSTANCE = String(Math.randInt(1000) + 1);
var DOWNLOAD;
var SELECTING = false; // toggle wether we are selecting things RN or not
var SELECT_TO = false;
// -------------------------------------------
// ----CANVAS CONFIGURATION ------------------
// -------------------------------------------

ctx = canv.getContext('2d');        //main canvas (z0)
ctxo = canvOverlay.getContext('2d'); //canvas for overlays (z1) (dynamic)
ctxbg = canvBg.getContext('2d');      //canvas for background drawing (z-1)

const toolBarHeight = 65 + 30;
const width = 800;
const height = 600;
const dpi = 2; //needs to be changed to dynamic dpi setup

//setting resolution
setupCanv = () =>
    [canvBg, canv, canvOverlay].map(curr => {
        curr.height = (innerHeight - toolBarHeight) * dpi;
        curr.style.height = (innerHeight - toolBarHeight);
        curr.width = (innerWidth - 2) * dpi;
        curr.style.width = innerWidth - 2;
        curr.getContext('2d').scale(dpi, dpi);
    });
setupCanv();


var MODE = 'Place';
var SEL_MODE = 'Subgraph';
var CONNECTION_MODE = 'Manual';
var NODE_WEIGHT_MODE = 'Manual';
var CONNECTION_SEL_MODE = 'Default';

var graph = new Graph();
var hist = new History();
var isMouseDown = false;        //checking mouseDown-ness
var lastClickedElement = null;  //last 'down' clicked element [null,Node]
var drawingLine = null;         //are we drawing a line? [Boolean]
var nodeNumber = 0;
var downCoords = [];
var moveCoords = [0, 0];
var translateCoords = [0, 0];
var canvBgColor = '#ffffff';
var textExtra = ['', '', ''];  // extra pre and after cursor text
var ConnectionHighlight = [];


function switchMode(mode) {
    ConnectionHighlight = [];


    MODE = mode;
    //console.log(mode);
    $('body').classList.remove('move-mode', 'search-mode', 'place-mode');
    switch (MODE) {
        case 'Move' :
            $('body').classList.add('move-mode');
            break;
        case 'Place' :
            $('body').classList.add('place-mode');
            break;
        case 'Search' :
            $('body').classList.add('search-mode');
            $('#line').classList.remove('on');
            break;
    }
}

function translateRender(type, e) {
    //translating everything
    //used for moving
    let x = e.offsetX;
    let y = e.offsetY;
    //console.log(e,type,x,y);

    if (type == 'click') {
        downCoords = [x, y];
        moveCoords = [0, 0];
    }

    ctx.translate(moveCoords[0], moveCoords[1]);
    ctxbg.translate(moveCoords[0], moveCoords[1]);
    ctxo.translate(moveCoords[0], moveCoords[1]);

    moveCoords = [downCoords[0] - x, downCoords[1] - y];
    //console.log(moveCoords,x,y,downCoords);

    ctx.translate(-moveCoords[0], -moveCoords[1]);
    ctxbg.translate(-moveCoords[0], -moveCoords[1]);
    ctxo.translate(-moveCoords[0], -moveCoords[1]);

    if (type == 'up') {
        translateCoords[0] += moveCoords[0];
        translateCoords[1] += moveCoords[1];
    }
}

//['green','blue','yellow','orange','magenta','brown','silver','salmon','fuschia','#cc6622','#66cc22']
const cols = MAT_COLORS.get.byHue(500);

Map.prototype.type = 'Map';

// addon sidebar can be disabled
sidebar = {
    showEl: () => false,

};

function deSelectRender(e) {
    ConnectionHighlight = [];
    if (e.target.id != 'canv') {
        render(null, {which: 3});
        //$('#properties').innerHTML ='';
    }

    // which 3 is right  click
}

//-------------------------------------


function updateSelMode(mode) {
    sidebar.highlightChanged(true)
    SEL_MODE = mode;
    render(null, {which: 3});
}

function updateConnectionMode(mode) {
    sidebar.highlightChanged(true)
    CONNECTION_MODE = mode;
    render(null, {which: 3});
}

function updateNodeConnectionMode(mode) {
    sidebar.highlightChanged(true)
    NODE_WEIGHT_MODE = mode;
    render(null, {which: 3});
}

function updateNodeConnectionHighlight(mode) {
    sidebar.highlightChanged(true)
    CONNECTION_SEL_MODE = mode;
    render(null, {which: 3});
}

function changeStruct(calle) {
    //gets called whenever the structure is to be changed ,

    //refreshing tree status
    circle.calculate();


    store.save();
}


function renderSelectedColor(val) {
    let mainColor = '#444';

    if (SEL_MODE == "Subgraph") {
        if (val.depth != undefined && treeColouring.value == "fromDST" && !circle.subGraphHas.has(val.subgraph)) {
            mainColor = MAT_COLORS.get.byColor(
                MAT_COLORS.swatches[val.subgraph % 10]
            )[(val.depth > 9) ? 0 : 9 - val.depth]
        } else {
            if (val.subgraph == null) mainColor = '#444'
            else mainColor = cols[val.subgraph % 10];
        }


    } else if (SEL_MODE == "NodeWeight") {

        mainColor = 'hsl(' + (Number(val.weight) * 2 + 200) + ',' + (Number(val.weight) + 50) + '%,60%)';

    } else if (SEL_MODE == "NodeConnection") {
        let acc = 0;
        val.edges.forEach((val, key) => {
            acc += val;
        });
        mainColor = 'hsl(' + (Number(acc) * 3 + 100) + ',' + (Number(acc) / 2 + 50) + '%,60%)';

    } else if (SEL_MODE == "Random") {
        let o = val.id;
        let O = 256;
        mainColor = new Color(nxt.seed(o, O), nxt.seed(o, O, 5), nxt.seed(o, O, 7)).rgb;

    }


    return mainColor;
}


randomName = () => Math.random().toString(36).replace(/[^a-z]+/g, '').replace(0, 5)

// returns:
// 1 if L*2 > r
// -1 if L < r
// 0 if not in node but in invalid location
function distance(x1, x2, y1, y2, r, reqDist) {
    let dist = Math.sqrt(Math.pow((x1 - x2), 2) + Math.pow((y1 - y2), 2)); //pithagoras

    if (r === true) return dist;
    if (dist > r * 2)
        return 1 //you can draw
    else if (dist > r)
        return 0 //intersect
    else
        return -1 //clicked at node
}

function setProps(el) {

    if (el) {
        const proplist = el.name.props?.map(prop => `<tr>
                <td>${prop[0]}</td>
                <td>${prop[1]}</td>
            </tr>`)
        const edges = [...el.edges].map(edge => `<Button className="btn" onClick="setProps(_gid(${edge[0]}))">${edge[0]}</Button>`)
        console.log(el.name.props, proplist)
        props_info.innerHTML = `
        <table>
        <tr><td colspan="2"><b>General</b></td> </tr>
        <tr><td>ID</td><td>${el.id}</td></tr>
        <tr><td>coordinates</td><td>${el.name.x}x - ${el.name.y}y</td></tr>
        <tr><td>text</td><td>${el.name.text ?? "-"}</td></tr>
        <tr><td colspan="2"><b>Properties</b></td> </tr>
        ${proplist}
        <tr><td colspan="2"><b>Edges From this</b></td> </tr>
        <tr><td colspan="2">${edges}</td></tr>
        </table>
        `
    }
}

_gid = (id) => graph.nodes.get(id)