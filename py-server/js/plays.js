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

function createChartDirective(directiveName, toggleVar, renderChartFn) {
  var containerName = directiveName+'Container'
  return { 
		replace: true,
		/*scope: {
		  items: '='
		  //charData : '='
		},*/
		controller: function ($scope, $element, $attrs) {
		  console.log(directiveName + ' controller...');
		},
		template: '<div id="'+containerName+'" style="margin: 0 auto">not working</div>',
		link: function (scope, element, attr) {
		  scope.$watch(toggleVar, function (newValue) {
			  if(scope[toggleVar]===1) {
			    renderChartFn(scope, containerName);
			  }
			}, true);
    }
  };
}

shPlays.directive('characterChart', function($document) {
  var directiveName = 'characterChart';
  var toggleVar = 'showCurrPlayChart';
  return createChartDirective(directiveName, toggleVar, function(scope, containerName) {
     createSinglePlayCharChart(scope, containerName);
  }); 
});

shPlays.directive('allPlaysChart', function($document) {
  var directiveName = 'allPlaysChart';
  var toggleVar = 'showAllPlaysChart';
  return createChartDirective(directiveName, toggleVar, function(scope, containerName) {
     createAllPlaysCharChart(scope, containerName);
  }); 
});

function createSinglePlayCharChart(scope, containerName) {
  console.log('Rendering Single Play...');

  var charMap = scope.charData;
  var characters = _.sortBy(scope.characters, function(c) { 
    return -1*charMap[c].nlines; 
  });
  var charlines = characters.map(function(c) {
    return charMap[c].nlines;
  });
  
	var chart = new Highcharts.Chart({
		chart: {
			type : 'column',
			renderTo : containerName,
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


function createAllPlaysCharChart(scope, containerName) {
  console.log('Rendering All Plays 1...');
  var allPlaysData = scope.allPlaysData;
  var playChars = {};
  
  var playLines = Object.keys(allPlaysData).map(function(playAlias) {
    var playCharacters = allPlaysData[playAlias];
    
    var characters = _.sortBy(Object.keys(playCharacters), function(characterName) { 
      return -1*playCharacters[characterName].nlines; 
    });
    playChars[playAlias] = characters; 
        
    return {
      name : playAlias,
	    data : characters.map(function(characterName, idx) {
        //return [idx, Math.log(playCharacters[characterName].nlines)];
        return [idx, playCharacters[characterName].nlines];
	    })
	  };
  });
  
  var chart = new  Highcharts.Chart({
    chart: {
		  type: 'spline',
		  renderTo : containerName,
		  //zoomType: 'x',
		},
		title: {
		  text: 'Plays / Characters'
		},
		subtitle: {
		  text: 'Irregular time data in Highcharts JS'
		},
		xAxis: {
      /*type: 'datetime',
                dateTimeLabelFormats: { // don't display the dummy year
                    month: '%e. %b',
                    year: '%b'
                }*/		
			type: 'int',
			max : 10
		},
		yAxis: {
			title: {
			  text: 'Total number of lines'
			},
			min: 0,
			//type: 'logarithmic',
      //minorTickInterval: 0.1
			//max : 1700
		},
		tooltip: {
			formatter: function() {
		    return '<b>'+ this.series.name +'</b><br/>'
		    //+Highcharts.dateFormat('%e. %b', this.x) +': '+ this.y +' m';
		    ;
			}
		},
		series: playLines
  });
  
  //chart.yAxis[0].setExtremes(500, 1700);
}
