var shPlays = angular.module('shPlays', [], function($locationProvider) {
  $locationProvider.html5Mode(true);
  //angular.bootstrap(document.documentElement);
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
  shPlays.directive(directiveName, function($document) {
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
			  
			  function refreshEvt(newValue) {
				  if(scope[toggleVar]!==0) {
				    renderChartFn(scope, containerName, scope[toggleVar]);
				  }
				};
			  scope.$watch(toggleVar, refreshEvt, true);
        // better way of doing this? 
			  scope.$watch('refreshToggle', refreshEvt, true);
	    }
	  };
	});
};

createChartDirective('characterChart', 'showCurrPlayChart', function(scope, containerName, whichMetric) {
  var charMap = scope.charData;
  var characters = scope.characters;
  
  var initChartFn = wu.curry(createSinglePlayCharChart, charMap, characters, containerName);
  switch(whichMetric) {
    case 'By Lines': initChartFn('nlines', 'Lines'); break;
    case 'By Edges': initChartFn('nedges', 'Edges'); break;
  };
});

createChartDirective('allPlaysChart', 'whichChart', function(scope, containerName, whichChart) {
  
  var allPlaysData = scope.allPlaysData;
  function isActivePlay(playAlias) {
    var playData = allPlaysData[playAlias];
    return (playData.genre=='Comedy' && scope.genreComedy) 
        || (playData.genre=='History' && scope.genreHistory)
        || (playData.genre=='Tragedy' && scope.genreTragedy);
  };

  var initColChart = wu.curry(createAllPlaysCharChart, allPlaysData, containerName, isActivePlay);
  var initSplineChart = wu.curry(createAllPlaysSplineChart, allPlaysData, containerName, isActivePlay);

  switch(whichChart) {
    case 'Top Characters (Lines)' : initColChart('nlines', '% of Total Lines'); break;
    case 'Top Characters (Edges)' : initColChart('nedges', '% of Total Edges'); break;
    case 'All Plays (Lines)'      : initSplineChart('nlines', '% of Total Lines'); break;
    case 'All Plays (Edges)'      : initSplineChart('nedges', '% of Total Edges'); break;
  };
});

function createSinglePlayCharChart(charMap, characters, containerName, whichMetric) {
  console.log('Rendering Single Play...');

  var characters = _.sortBy(characters, function(c) { 
    return -1*charMap[c][whichMetric]; 
  });
  var charAggr = characters.map(function(c) {
    return charMap[c][whichMetric];
  });
  
  createColumnChart(containerName, characters, charAggr, 'Characters');
}

function mapAllPlaysByMetric(allPlaysData, whichMetric) {
  var playChars = {};
  return Object.keys(allPlaysData).map(function(playAlias) {
    var playCharacters = allPlaysData[playAlias].chardata;
    var nTotal = 0; 
    var characters = _.sortBy(Object.keys(playCharacters), function(characterName) {
      var charAggr = playCharacters[characterName][whichMetric];
      nTotal += charAggr;
      return -1*charAggr; 
    });
    playChars[playAlias] = characters; 

    return {
      name : playAlias,
	    data : characters.map(function(characterName, idx) {
        //return [idx, Math.log(playCharacters[characterName].nlines)];
        return [idx+1, playCharacters[characterName][whichMetric] / nTotal, characterName];
	    })
	  };
  });
} 

function mapAllChars(allPlaysData, mapByField) {
  var playCharAggr = {};
  
  $.each(Object.keys(allPlaysData), function(idx, playAlias) {
    var playCharacters = allPlaysData[playAlias].chardata;
    var nTotal = 0; 
    
    $.each(Object.keys(playCharacters), function(idx, characterName) {
      var charAlias = playAlias+'/'+characterName;
      playCharAggr[charAlias] = playCharacters[characterName][mapByField];
      nTotal += playCharacters[characterName][mapByField];
    });
    
    $.each(Object.keys(playCharacters), function(idx, characterName) {
      var charAlias = playAlias+'/'+characterName;
      playCharAggr[charAlias] = playCharAggr[charAlias] / nTotal;
    });
  });
  
  var characters = _.sortBy(Object.keys(playCharAggr), function(charAlias) {
	  return -1*playCharAggr[charAlias]; 
	});
  
	return {
	  characters : characters, 
	  fieldAggr : playCharAggr
  };
}

function createAllPlaysCharChart(allPlaysData, containerName, isActivePlay, whichMetric, Label) {
  console.log('Rendering All Plays 1...');
  
  var playCharAggr = mapAllChars(allPlaysData, whichMetric);
  var characters = playCharAggr.characters.slice(0, 30);
  
  var aggrData = characters.map(function(c) {
    return playCharAggr.fieldAggr[c];
  });
  
  createColumnChart(containerName, characters, aggrData, 'Top Characters', Label);
};

function createColumnChart(containerName, categories, singleSeries, title, YAxisLabel) {
  if(!YAxisLabel)
    YAxisLabel = 'Lines';
	
	var chart = new Highcharts.Chart({
		chart: {
			type : 'column',
			renderTo : containerName,
		},
		title: {
		  text: title
		},
    xAxis: {
      categories : categories,
      labels : {
        rotation: -45,
        align : 'right'
      },
    },
		yAxis: {
			min: 0,
			title: {
		    text: YAxisLabel
			}
		},
		legend: {
		  enabled: false
		},
		tooltip: {
			formatter: function() {
		    return '<b>'+ this.key +'</b><br/>' + Math.round(this.y*10000)/100 + '%'
		    //+Highcharts.dateFormat('%e. %b', this.x) +': '+ this.y +' m';
		    ;
			}
		},
		plotOptions: {
			column: {
				//pointPadding: 0.2,
				borderWidth: 0
			}
		},
		series: [ { name : 'A', data : singleSeries } ]
	});
}

function createAllPlaysSplineChart(allPlaysData, containerName, isActivePlay, whichMetric, label) {
  
  var playAggrData = mapAllPlaysByMetric(allPlaysData, whichMetric).map(function(playData) {
    if(!isActivePlay(playData.name))
      playData.visible = false;
    return playData;
  });

  var charNames = {};
  playAggrData.map(function(playData) {
    var playName = playData.name;
    charNames[playName] = playData.data.map(function(charData) {
      return charData[2]; // characterName
    })
  });
  
  var chart = new  Highcharts.Chart({
    chart: {
		  type: 'spline',
		  renderTo : containerName,
		  zoomType: 'x',
		  height : '650'
		},
		title: {
		  text: 'Plays / Characters'
		},
		//subtitle: { text: 'Irregular time data in Highcharts JS' },
    xAxis: {
      type: 'int',
			min : 1,
			max : 10
    },
		yAxis: {
			title: {
			  text: label
			},
			min: 0,
			//type: 'logarithmic',
      //minorTickInterval: 0.1
			//max : 1700
		},
		tooltip: {
			formatter: function() {
			  var playAlias = this.series.name;
			  var charIdx = this.key;
			  var charName = charNames[playAlias][charIdx-1];
			
		    return '<b>'+ playAlias +'</b><br/>' +
		           '<b>'+ charName +'</b><br/>' + Math.round(this.y*10000)/100 + '%'
		    //+Highcharts.dateFormat('%e. %b', this.x) +': '+ this.y +' m';
		    ;
			}
		},
		series: playAggrData
  });
  //chart.yAxis[0].setExtremes(500, 1700);
}  
