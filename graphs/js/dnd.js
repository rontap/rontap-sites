// drag and drop support
var FORCE_DND_END = false;
function dndFactory(el) {


  window[el].onmousedown = function(event) {
  window[el].style.zIndex = 900;
  if (event.target.id == el+'Drag') {
  let shiftX = event.clientX - window[el].getBoundingClientRect().left;
  let shiftY = event.clientY - window[el].getBoundingClientRect().top;

  window[el].style.position = 'absolute';

  document.body.append(window[el]);

  moveAt(event.pageX, event.pageY);

  // centers the ball at (pageX, pageY) coordinates
  function moveAt(pageX, pageY) {
    if (FORCE_DND_END) {
      stopMove(true);
      FORCE_DND_END=false;
    }
    let offHeight = event.target.parentElement.offsetHeight;
    if (pageX>260 && pageX < innerWidth-30) {
      window[el].style.left = pageX - shiftX + 'px';

    }
    if ( pageY>120 && pageY < innerHeight-offHeight+20) {
      window[el].style.top = pageY - shiftY + 'px';
    }


  }

  function onMouseMove(event) {
    moveAt(event.pageX, event.pageY);
  }

  // (3) move the ball on mousemove
  document.addEventListener('mousemove', onMouseMove);


  function stopMove(event) {

    document.removeEventListener('mousemove', onMouseMove);
    window[el].onmouseup = null;

    if (event==false)
      if (['lineDrag','line','linkRef'].indexOf(event.srcElement.id) !== -1)
      window[el].style.zIndex = 800;
      else
      window[el].style.zIndex = 810;

  };
  document.body.onmouseup = function(event){stopMove(event)};

};

window[el].ondragstart = function() {
  return false;
};

}
}


//assigning factory
['sidebarProp','line','action','distance','viewer','textStyle','multiselect','treeObj','props'].map( (x) => dndFactory(x));


minimiseFactory =(el) => {
  console.log(">Minimising " +el);
  $('#'+el).classList.remove('open','on');
  name = $('#'+el+' h2').innerHTML;
  footerBarHolder.innerHTML+=`<div linked=${el} onclick="$('#${el}').classList.add('open','on');this.parentNode.removeChild(this);">${name.replace(/(<(.*?)>)+/g,'')}</div>`

}
openWindow = (el) => {
  if ($$('[linked="'+el+'"]').length != 0 ) {
    //$('[linked="'+el+'"]').onclick();
  }
  else {
    $('#'+el).classList.add('on','open');
  }
}
closeWindow = (el) => {
  if ($$('[linked="'+el+'"]').length != 0 ) {
    $('[linked="'+el+'"]').onclick();

  }
    $('#'+el).classList.remove('on','open');
}


openWindow('viewer');
minimiseFactory('viewer');
openWindow('textStyle');
minimiseFactory('textStyle');

openWindow('multiselect');

minimiseFactory('multiselect');

openWindow('treeObj');
minimiseFactory('treeObj');
openWindow('props')
// minimiseFactory('props')
//

