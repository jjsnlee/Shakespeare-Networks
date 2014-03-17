var shPlays = angular.module('shPlays', [], function($locationProvider) {
  $locationProvider.html5Mode(true);
});

shPlays.directive('myDraggable', function($document) {
  // Example from: http://docs.angularjs.org/guide/directive
  return function(scope, element, attr) {
    addDnD($document, scope, element, attr);
  };
});

function addDnD(document, scope, element, attr) {
	var divElmt = element[0]; // the div tag
	// get the actual position of the image 

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

shPlays.directive('characterChart', function($document) {
  return { 
		replace: true,
		/*scope: {
		  items: '='
		  //char_data : '='
		},*/
		controller: function ($scope, $element, $attrs) {
		  console.log('Controller...');
		},
		template: '<div id="container" style="margin: 0 auto">not working</div>',
		link: function (scope, element, attr) {
		  scope.$watch("showChart", function (newValue) {
			  if(scope.showChart===1) {
			    createChart(scope);
			  }
			}, true);
    }
  };
});

function createChart(scope) {
  console.log('Rendering 1...');

  var charMap = scope.play_content.char_data;  
  var characters = _.sortBy(scope.play_content.characters, function(c) { 
    return -1*charMap[c].nlines; 
  });
  var charlines = characters.map(function(c) {
    return charMap[c].nlines;
  });
  
	var chart = new Highcharts.Chart({
		chart: {
			type : 'column',
			renderTo : 'container',
		},
		title: {
		  text: 'Characters'
		},
    xAxis: {
      categories : characters,
      labels : { 
        rotation: -45,
        align : 'right' 
      }
    },
		yAxis: {
			min: 0,
			title: {
		    text: 'Lines'
			}
		},
		legend: {
		  enabled: false
		},
		plotOptions: {
			column: {
				pointPadding: 0.2,
				borderWidth: 0
			}
		},
		series: [ { name : 'A', data : charlines } ]
	});  
}

