// Example from: http://docs.angularjs.org/guide/directive
function addDnD(document, scope, element, attr) {
	var divElmt = element[0]; // the div tag
	// get the actual position of the image 

  // Keep track of the positions of each of the popups.  
  // Otherwise there will be some jumping of the div elements
  scope.popupPosn = {};
  
	var popupPosn = scope.popupPosn[divElmt.id] = {
	  startX : divElmt.offsetLeft, 
	  startY : divElmt.offsetTop
	};
	popupPosn.x = popupPosn.startX;
	popupPosn.y = popupPosn.startY;
	
	element.css({
	  //position: 'relative',
	  //border: '1px solid red',
	  //backgroundColor: 'lightgrey',
	  cursor: 'pointer'
	});
 
	element.on('mousedown', function(event) {
	  // Prevent default dragging of selected content
	  event.preventDefault();
	  popupPosn.startX = event.pageX - popupPosn.x;
	  popupPosn.startY = event.pageY - popupPosn.y;
	  document.on('mousemove', mousemove);
	  document.on('mouseup', mouseup);
	});

	function mousemove(event) {
	  popupPosn.y = event.pageY - popupPosn.startY;
	  popupPosn.x = event.pageX - popupPosn.startX;
	  element.css({
	    top:  popupPosn.y + 'px',
	    left: popupPosn.x + 'px'
	  });
	}

	function mouseup() {
	  document.unbind('mousemove', mousemove);
	  document.unbind('mouseup', mouseup);
	}
};

