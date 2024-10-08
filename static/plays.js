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
  shPlays.directive(directiveName, function($document, $compile) {
    var containerName = directiveName+'Container'
    return { 
      replace: true,
      /*scope: { items: '=' },*/
      controller: function ($scope, $element, $attrs) {
        console.log(directiveName + ' controller...');
      },
      template: '<div id="'+containerName+'" style="margin: 0 auto">Loading...</div>',
      link: function (scope, element, attr) {
        function refreshEvt(newValue) {
          if(scope[toggleVar]!==0) {
            renderChartFn(scope, containerName, scope[toggleVar], $compile);
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

createChartDirective('allPlaysChart', 'whichChart', function(scope, containerName, whichChart, compile) {
  
  var allPlaysData = scope.allPlaysSceneSummary;
  var isPlayActive = scope.isPlayActive;
  
  var initSummary = wu.curry(createAllPlaysOneSummary, allPlaysData, containerName, isPlayActive);
  var initColChart = wu.curry(createAllPlaysCharChart, allPlaysData, containerName, isPlayActive);
  var initSplineChart = wu.curry(createAllPlaysSplineChart, allPlaysData, containerName, isPlayActive);

  switch(whichChart) {
  	case 'All Plays:Summary'   : initSummary(); break;
  	// not sure if this needs to change for the isPlayActive call
  	case 'All Plays:Degrees'       : initDegreeChart(scope, compile, allPlaysData, containerName); break;
  	case 'All Plays:Densest'       : initDensityChart(scope, compile, allPlaysData, containerName); break;
  	
    case 'Top Characters (Lines)'  : initColChart('nlines', '% of Total Lines'); break;
    case 'Top Characters (Edges)'  : initColChart('degrees', '% of Total Edges'); break;
    case 'All Plays (Lines)'       : initSplineChart('nlines', '% of Total Lines'); break;
    case 'All Plays (Edges)'       : initSplineChart('degrees', '% of Total Edges'); break;
  };
});

function createAllPlaysOneSummary(allPlaysData, containerName, isPlayActive) {
  console.log('allPlaysData: '+allPlaysData);
  var playsPerRow = 4;
  var currRow=0, currCol=0;
  var templ = '<table width="100%" border="1" cellpadding="0" cellspacing="0">';

  var activePlaysKeys = wu(Object.keys(allPlaysData)).filter(function(playAlias) {
    return isPlayActive(playAlias)
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

function mapAllPlaysByCharMetric(allPlaysData, whichMetric) {
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

function createAllPlaysCharChart(allPlaysData, containerName, isPlayActive, whichMetric, Label) {
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
	);
}

function createAllPlaysSplineChart(allPlaysData, containerName, isPlayActive, whichMetric, label) {
  
  var playAggrData = mapAllPlaysByCharMetric(allPlaysData, whichMetric).map(function(playData) {
    if(!isPlayActive(playData.name))
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

function initDensityChart(scope, compile, allPlaysData, containerName) {
	var allPlays = Object.keys(allPlaysData);
	allPlays.sort();
  var templ = '<div style="width:600px;overflow:none"><table border=1 cellpadding=1 cellspacing=1>'
	$.each(allPlays, function(idx, playName) {
		var play = allPlaysData[playName];
		if(scope.isPlayActive(playName)) {
			templ += '<tr><td style="font-size:10pt">'+play.title+'</td>';
			$.each(play.scenes, function(idx, scene) {
				if(scene.total_degrees > 10) {
					templ += '<td><img src="/' + scene.graph_img_f + '" style="height:200px" '+
						'ng-click="sceneOpen($event,\''+ playName+'\','+idx+')"/></td>';
				}
			});
			templ += '</tr>';
		}
  });
  templ += '</table></div>';
	
	/*
	var col1 = col2 = '';
  $.each(allPlays, function(idx, playName) {
		var play = allPlaysData[playName];
		if(scope.isPlayActive(playName)) {
			col1 += '<tr height="120"><td style="font-size:10pt">'+play.title+'</td></tr>';
			col2 += '<tr height="120">';
			$.each(play.scenes, function(idx, scene) {
				if(scene.total_degrees > 10) {
					col2 += '<td><img src="/' + scene.graph_img_f + '" height="23%" '+
						'ng-click="sceneOpen($event,\''+ playName+'\','+idx+')"/></td>';
				}
			});
			col2 += '</tr>';
		}
  });
  var templ = '<div style="overflow:none"><table border=0 cellpadding=0 cellspacing=0><tr valign="top"><td>';
  templ += '<table border=1 cellpadding=1 cellspacing=1>'+col1+'</table>';
  templ += '</td><td>';
  templ += '<div style="width:600px;overflow:none"><table border=1 cellpadding=1 cellspacing=1>'+col2+'</table></div>';
  templ += '</td></tr></table></div>';
	*/
  
	//templ += '<div id="innerContainer"></div>'
	$('div#'+containerName).html(compile(templ)(scope));
}

function initDegreeChart(scope, compile, allPlaysData, containerName) {
	if(!scope.selectedDegreeYAxis)
		scope.selectedDegreeYAxis = 'Degree Assortative Coefficient';

	var allPlays = Object.keys(allPlaysData);
	allPlays.sort();
	
	var ySets = {
		'Avg Clustering' : {
			y : function(v) { return v.sc_avg_clustering },
			ignoreIf : function(val) { return val <= -1; }
		},
		'Avg Shortest Path' : {
			y : function(v) { return v.sc_avg_shortest_path },
			ignoreIf : function(val) { return val <= -1; }
		},
		'Degree Assortative Coefficient' : {
			y : function(v) { return v.sc_deg_assort_coeff },
		},
		'Plays By Title' : {
			y : function(v) { return v.play_idx },
			yAxis : {
				categories: allPlays,
				type: 'category',
				//tickInterval: null,
				//tickPositions: _.map(new Array(allPlays.length), function(v,i) { return i; }),
				//tickPositions: Array.apply(null, {length: allPlays.length}).map(Number.call, Number),
				min: 0, 
				max: allPlays.length-1,
			}
		} 
	};
	
	var allPlaysPairs = _.pairs(allPlaysData);
	allPlaysPairs.sort(function(a,b) { return a[1].genre.localeCompare(b[1].genre) });
	var allPlaysByGenre = allPlaysPairs.map(function(d) { return d[0]; })
	ySets['Play Grouped by Genre'] = $.extend(true, {}, ySets['Plays By Title']);
	ySets['Play Grouped by Genre'].yAxis.categories = allPlaysByGenre;
	ySets['Play Grouped by Genre'].playsOrder = allPlaysByGenre;

	allPlaysPairs.sort(function(a,b) { return a[1].year.localeCompare(b[1].year) });
	var allPlaysByYear = allPlaysPairs.map(function(d) { return d[0]; })
	ySets['Play Grouped by Year'] = $.extend(true, {}, ySets['Plays By Title']);
	ySets['Play Grouped by Year'].yAxis.categories = allPlaysByYear;
	ySets['Play Grouped by Year'].playsOrder = allPlaysByYear;
	
	scope.degreeMin = 25;
	scope._degreeYAxisOptions = Object.keys(ySets);
  var templ = 'y-axis: <select name="degreeYAxis" ng-model="selectedDegreeYAxis" '
      + 'ng-options="p for p in _degreeYAxisOptions" '
      + 'ng-change="renderDegreeChart()" style="width:200px"></select>'
      + '&nbsp;Min # of Degrees: <input type="text" id="min-degree" ng-model="degreeMin" ng-blur="renderDegreeChart()" >' // style="border:1; color:#f6931f; font-weight:bold;"
      //+ '<input type="text" id="max-degree" ng-model="degreeMax" readonly style="border:0; color:#f6931f; font-weight:bold;">'
      //+ '<div id="slider-range-max"></div><br>'
      //+ '<script>$(function() { enableSliderWidget("#min-degree") } );</script>'
      + '<br>';

  templ += '<div id="innerContainer"></div>'
  $('div#'+containerName).html(compile(templ)(scope));
  
  //var DEGREE_THRESHOLD = 25;
  var playsVisibility = {};

	scope.renderDegreeChart = function() {
  	var whichMetric = 'total_degrees'
  	var whichY = scope.selectedDegreeYAxis;
  	var ySet = ySets[whichY];
  	
  	var plays = allPlays;
  	if(ySet.playsOrder)
  		plays = ySet.playsOrder;
  	
    function mapAllPlaysBySceneMetric(whichMetric) {
      return plays.map(function(playAlias, playIdx) {
        var play = allPlaysData[playAlias];
        var scenes = play.scenes;
        var series = [];
        _.each(scenes, function(scene, _idx) {
        	
        	var val = {
          	play : play.title,
          	play_idx : playIdx,
          	year : play.year,
          	genre : play.genre,
            name : scene.scene,
            dataLabels : {
              formatter: function() {
                return play.title + '<br>' +scene.scene
              },
            },
            x : scene.total_degrees, 
            z : Math.pow(scene.total_lines, 2),
            //z : scene.total_lines,

            sc_total_degrees : scene.total_degrees,
            sc_total_edges : -1,
            sc_total_lines : scene.total_lines,
            sc_avg_clustering : scene.avg_clustering,
            sc_avg_shortest_path : scene.avg_shortest_path,
            sc_deg_assort_coeff : scene.deg_assort_coeff,
            sc_closeness_vitality : scene.closeness_vitality,
            sc_density : scene.density,
            sc_location : scene.location,
            sc_graph_img_f : scene.graph_img_f  
          };
        	
        	val.y = ySet.y(val);
        	var ignoreIf = ySet.ignoreIf || function(v) { return false; };
        	if(ignoreIf(val) || scene.total_degrees < scope.degreeMin)
        		return;
        	
          series.push(val);
        });
        
        return { name : playAlias, data : series };
      });
    };
    
    var playAggrData = mapAllPlaysBySceneMetric(whichMetric).map(function(playData) {
    	if(playsVisibility[playData.name]) 
    		playData.visible = playsVisibility[playData.name];
    	else if(!scope.isPlayActive(playData.name) ) //|| playData.name!='hamlet'
        playData.visible = false;
      return playData;
    });
    
    var yAxis = { title: { text: whichY } };
    if(ySet.yAxis)
    	yAxis = $.extend(yAxis, ySet.yAxis);

  	var chart = new Highcharts.Chart({
			chart: {
		    type: 'bubble',
		    //renderTo : containerName,
		    renderTo : 'innerContainer',
		    zoomType: 'xy',
		    height: 700,
		    //events: { load: function () { chart.hideLoading(); } }
			},
	    plotOptions: {
	      series: {
	        dataLabels: {
	          enabled: true,
	          //borderRadius: 5, 
	          backgroundColor: 'rgba(252, 255, 197, 0.3)',
	          borderWidth: 1,
	          borderColor: '#AAA',
	          x: -5, y: -6
	        },
	        events: {
	  		    legendItemClick: function () {
	  		    	// preserve the selected plays when toggling between options
	  		    	playsVisibility[this.name] = !this.visible;
	  		    }
	        }
	      },
	    },
	    tooltip: {
	    	useHTML: true,
	    	hideDelay: 200,
	      formatter: function() {
	      	var pt = this.point;
	      	return '<b>' + pt.play + ' ' + pt.name + '</b><br>'
	      				+ 'Year:' + pt.year + '<br>'
	      				+ 'Scene location: ' + pt.sc_location + '<br>'
	      				+ 'Total Degrees: ' + pt.sc_total_degrees + '<br>'
	      				+ 'Total Lines: ' + pt.sc_total_lines + '<br>'
	      				
	      				+ 'Avg Clustering:' + pt.sc_avg_clustering.toFixed(4) + '<br>'
	      				+ 'Avg Shortest Path:' + pt.sc_avg_shortest_path.toFixed(4) + '<br>'
	      				+ 'Closeness Vitality:' + pt.sc_closeness_vitality + '<br>'
	      				+ 'Degree Assortativity Coefficient:' + pt.sc_deg_assort_coeff.toFixed(4) + '<br>'
	      				+ 'Density:' + pt.sc_density.toFixed(4) + '<br>'
	      				+ '<img src="/' + pt.sc_graph_img_f + '" height="23%"/>'  
	      }
	    },
	    title: { text: 'Degrees v ' + whichY },
			xAxis: { title: { text: 'Total Degrees' } },
			yAxis: yAxis,
			series: playAggrData
		});
  	//chart.showLoading();
  };
  
  scope.renderDegreeChart();
}
