var shPlays = angular.module('shPlays', [], function($locationProvider) {
  $locationProvider.html5Mode(true);
  //angular.bootstrap(document.documentElement);
});

shPlays.directive('myDraggable', function($document) {
  return function(scope, element, attr) {
    addDnD($document, scope, element, attr);
  };
});

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
      template: '<div id="'+containerName+'" style="margin: 0 auto">Loading...</div>',
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
    case 'By Edges': initChartFn('degrees', 'Edges'); break;
  };
});

createChartDirective('allPlaysChart', 'whichChart', function(scope, containerName, whichChart) {
  
  var allPlaysData = scope.allPlaysSceneSummary;
  function isActivePlay(playAlias) {
    var playData = allPlaysData[playAlias];
    return (playData.genre=='Comedy' && scope.genreComedy) 
        || (playData.genre=='History' && scope.genreHistory)
        || (playData.genre=='Tragedy' && scope.genreTragedy);
  };

  var initSummary = wu.curry(createAllPlaysOneSummary, allPlaysData, containerName, isActivePlay);
  var initColChart = wu.curry(createAllPlaysCharChart, allPlaysData, containerName, isActivePlay);
  var initSplineChart = wu.curry(createAllPlaysSplineChart, allPlaysData, containerName, isActivePlay);

  switch(whichChart) {
  	case 'All Plays - One Summary' : initSummary(); break;
  	// not sure if this needs to change for the isActivePlay call
  	case 'All Plays - Degree'      : initDegreeChart(allPlaysData, containerName, isActivePlay); break;
    case 'Top Characters (Lines)'  : initColChart('nlines', '% of Total Lines'); break;
    case 'Top Characters (Edges)'  : initColChart('degrees', '% of Total Edges'); break;
    case 'All Plays (Lines)'       : initSplineChart('nlines', '% of Total Lines'); break;
    case 'All Plays (Edges)'       : initSplineChart('degrees', '% of Total Edges'); break;
  };
});

function createAllPlaysOneSummary(allPlaysData, containerName, isActivePlay) {
  console.log('allPlaysData: '+allPlaysData);
  var playsPerRow = 4;
  var currRow=0, currCol=0;
  var templ = '<table width="100%" border="1" cellpadding="0" cellspacing="0">';

  var activePlaysKeys = wu(Object.keys(allPlaysData)).filter(function(playAlias) {
    return isActivePlay(playAlias)
  }).toArray();
  
  $.each(activePlaysKeys, function(idx, playAlias) {
    if(idx % playsPerRow == 0) {
      if(idx>0)
        templ+='</tr>';
      templ+='<tr>';
    }
    var playContainerName = '_'+playAlias; 
    templ+='<td width="25%"><div style="height:200px" ng-click="testClick()" id="' 
      + playContainerName +'">' + playAlias + '</div></td>';
  });
  
  templ+='</table>';
  // odd, the previous chart div contents are being retained, what a mess...
  $('div#'+containerName).html(templ);
  
  $.each(activePlaysKeys, function(idx, playAlias) {
    var playData = allPlaysData[playAlias];
    var charMap = playData.chardata;
    var characters = Object.keys(charMap);
    var playContainerName = '_'+playAlias;
    //var title = '<a ng-click="testClick()>'+playAlias+'</a>'
    var title = '<span style="font-size: 11px;"><b>'+playData.title + '</b><br>('+ playData.year+')</span>';
    
    chartOptions = {
      xAxisLabels : { enabled: false },
      title : title,
      titleUseHTML : true
    };
    createSinglePlayCharChart(charMap, characters, playContainerName, 'nlines', chartOptions);
  });
}

function createSinglePlayCharChart(charMap, characters, containerName, whichMetric, chartOptions) {
  console.log('Rendering Single Play...');
  if(!chartOptions)
    chartOptions = {}
  if(!chartOptions.title)
    chartOptions.title = 'Characters';

  var characters = _.sortBy(characters, function(c) { 
    return -1*charMap[c][whichMetric]; 
  });
  var charAggr = characters.map(function(c) {
    return charMap[c][whichMetric];
  });
  
  var seriesTotal = wu(charAggr).reduce(function(n, m) { return n+m });
  chartOptions.tooltip = {
    formatter: function() {
      return '<b>'+ this.key +'</b><br/>' + this.y + ' of ' + seriesTotal +' lines (' + Math.round(100*this.y/seriesTotal)+'%)' 
    }
  };
  
  createColumnChart(containerName, characters, charAggr, chartOptions);
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
  var characters = playCharAggr.characters.slice(0, 70);
  
  var aggrData = characters.map(function(c) {
    return playCharAggr.fieldAggr[c];
  });
  
  chartOptions = {
    title : 'Top Characters', 
    YAxisLabel : Label 
  }
  
  createColumnChart(containerName, characters, aggrData, chartOptions);
};

function createColumnChart(containerName, categories, singleSeries, chartOptions) {

  if(!chartOptions)
    chartOptions = {};
	
  YAxisLabel = chartOptions.YAxisLabel;
  title = chartOptions.title;
  titleUseHTML = chartOptions.titleUseHTML;
  xAxisLabels = chartOptions.xAxisLabels;
  tooltip = chartOptions.tooltip;
	
	if(!tooltip)
		tooltip = {
			formatter: function() {
		    return '<b>'+ this.key +'</b><br/>' + Math.round(this.y*10000)/100 + '%';
			}
		};
	
	if(!YAxisLabel)
    YAxisLabel = 'Lines';

	if(!xAxisLabels)
  	xAxisLabels = {
			rotation: -45,
			align : 'right'
		};
	
	var chart = new Highcharts.Chart({
		chart: {
			type : 'column',
			renderTo : containerName,
		},
		title: {
		  text: title,
		  useHTML:titleUseHTML
		},
		xAxis: {
		  categories : categories,
		  labels : xAxisLabels
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
		tooltip: tooltip,

		plotOptions: {
			column: {
				//pointPadding: 0.2,
				borderWidth: 0
			}
		},
		series: [ { name : 'A', data : singleSeries } ]
	}
//	,function(chart){
//    $('.highcharts-title').click(function(){
//        console.log('aaaa');
//    });
//	}
	);
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
  
  var chart = new Highcharts.Chart({
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

function initDegreeChart(allPlaysData, containerName, isActivePlay) {
  var whichMetric = 'total_degrees'
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
	
	var chart = new Highcharts.Chart({
		chart: {
	    type: 'bubble',
	    renderTo : containerName,
	    zoomType: 'xy'
		},
		title: {
			text: 'Highcharts Bubbles'
		},
		series: [
//		{ data: [[97,36,79],[94,74,60],[68,76,58],[64,87,56],[68,27,73],[74,99,42],[7,93,87],[51,69,40],[38,23,33],[57,86,31]] }, 
//		{ data: [[25,10,87],[2,75,59],[11,54,8],[86,55,93],[5,3,58],[90,63,44],[91,33,17],[97,3,56],[15,67,48],[54,25,81]] }, 
//		{ data: [[47,47,21],[20,12,4],[6,76,91],[38,30,60],[57,98,64],[61,17,80],[83,60,13],[67,78,75],[64,12,10],[30,77,82]] }
		]
	});
}
