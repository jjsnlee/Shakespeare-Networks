var termiteTopics = angular.module('termiteTopicDisplay', [], function($locationProvider) {
  $locationProvider.html5Mode(true);
  //angular.bootstrap(document.documentElement);
});

termiteTopics.directive('myDraggable', function($document) {
  return function(scope, element, attr) {
    addDnD($document, scope, element, attr);
  };
});

//termiteTopics.factory('stateModel', function() {
//  return new StateModel();
//});

termiteTopics.service('termiteMsgService', function() {
  var topicHandler = null;
  var setTopic = function(LdaModel, topicIndex, topicLabel) {
		topicHandler(LdaModel, topicIndex, topicLabel);
  };
  var registerTopicHandler = function(handler) { topicHandler = handler; };

  return {
  	registerTopicHandler : registerTopicHandler,
  	setTopic : setTopic,
  };
});

termiteTopics.controller('contentCtrl', function($scope, $http, $sce, termiteMsgService) {
  $scope.topDocsForTopic = [];
  
  // As a string
  $scope.selectedTopic = null;
  // As an index
  var selectedTopicIndex = -1;
  
  $scope.showTopicDetails = 0;

  var models = $scope.$parent.getModels();
  var filteredTermTopicProbabilityModel = models['filteredTermTopicProbabilityModel'];
  var termFrequencyModel = models['termFrequencyModel'];
  var stateModel = models['stateModel'];
  
  $scope.colorAxis = {
    //colors: ['#', '#','#', '#', '#', '#'],
    //colors: ['#E7E0D9', '#E7CFB7','#E7B98A', '#FFCE9E', '#FFB164', '#FF7F00'],
    colors: ['#A89D7D', '#A89049','#E7AD00', '#FFB164', '#FFFF00', '#DFFF00'],
    values: [0, 1, 10, 25, 50, 100]
  };
  var scoreColor = function(score) {
    var colorIdx = 0;
    for(i in $scope.colorAxis.values) {
      if(score < $scope.colorAxis.values[i]) {
        break;
      }
      colorIdx = i;
    }
    return $scope.colorAxis.colors[colorIdx]; 
  }

  $scope.getDocContent = function(charNm) {
    console.log('charNm: '+charNm);
    $scope.showTopicDetails = 1;
    $http.get('/shakespeare/corpus/characters/'+charNm).success(function(data) {
      console.log('data: '+data);
      var termTopicScoreMatrix = filteredTermTopicProbabilityModel.get('matrix');
      var terms = filteredTermTopicProbabilityModel.get('termIndex');
      var content = data.doc_content.map(function(section) {
        var sectionText = section.map(function(li) {
          var wds = li.split(' ');
          var uniqueWds = {};
          _.each(wds, function(wd) {
            wd = wd.toLowerCase();
            if(wd in uniqueWds) 
              return;
            uniqueWds[wd] = 1;
            var idx = terms.indexOf(wd);
            if(idx>-1) {
              var termScore = termTopicScoreMatrix[idx][selectedTopicIndex];
              if(termScore) {
	              var heatMapColor = scoreColor(termScore);
	              li = li.replace(new RegExp('\\b('+wd+')\\b', 'gi'), 
	                    "<span style='font-weight:bold;background-color:"+heatMapColor+"'>$1</span>" );
	            }
            }
          });
          return li;
        }).join('<br>\n');
        
        return '<p class="charText">'+sectionText+'</p>'
      }).join('\n');
      
      content = $sce.trustAsHtml(content);
      $scope.docName    = data.doc_name;
      $scope.docContent = content;
    });
  };
  
  function getSelectedTopic(LdaModel, topicIndex, topicLabel) {
    console.log('Will fetch data for topicIndex: ' + topicIndex);
    $scope.selectedTopic = topicLabel;
    selectedTopicIndex = topicIndex;
    $http.get('/shakespeare/corpus/lda/'+LdaModel+'/'+topicIndex).success(function(data) {
      data = data.map(function(c) {
        return {
          'char'  : c[0], 
          'score' : c[1].toFixed(6)
          //'url'   : 'ABC'
        }
      });
      $scope.topDocsForTopic = _.sortBy(data, function(c) { 
        return -1*c.score; 
      });
    });
  };
  
  termiteMsgService.registerTopicHandler(getSelectedTopic);
  
  // probably a much better way to do this...
	if(stateModel.get('doubleClickTopic')) {
	  var topicIndex = stateModel.get('doubleClickTopic');
	  //var topicLabel = termTopicMatrixView.parentModel.attributes.topicIndex[topicIndex];
	  //var topicLabel = filteredTermTopicProbabilityModel.attributes.topicIndex[topicIndex];
	  var topicLabel = 'Topic '; //+(topicIndex+1);
	  getSelectedTopic($scope.$parent.LDAModel, topicIndex, topicLabel);
	}
});
