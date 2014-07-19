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
  setTopic = function(LdaModel, topicIndex, topicLabel) {
		topicHandler(LdaModel, topicIndex, topicLabel);
  };
  registerTopicHandler  = function(handler) { topicHandler = handler; };

  return {
  	registerTopicHandler : registerTopicHandler,
  	setTopic : setTopic,
  };
});

termiteTopics.controller('contentCtrl', function($scope, $http, $sce, termiteMsgService) {
  $scope.topDocsForTopic = [];
  $scope.selectedTopic = null;
  $scope.showTopicDetails = 0;

  var models = $scope.$parent.getModels();
  var filteredTermTopicProbabilityModel = models['filteredTermTopicProbabilityModel'];
  var termFrequencyModel = models['termFrequencyModel'];
  var stateModel = models['stateModel'];
  
  $scope.getDocContent = function(charNm) {
    console.log('charNm: '+charNm);
    $scope.showTopicDetails = 1;
    $http.get('/shakespeare/corpus/characters/'+charNm).success(function(data) {
      console.log('data: '+data);
      
      var terms = filteredTermTopicProbabilityModel.get('termIndex');
      var content = data.doc_content.map(function(section) {
        var sectionText = section.map(function(li) {
          //return li.replace(/France/g, "<span class='yellow'>France</span>");
          for(i in terms)           
            li = li.replace(new RegExp('\\b('+terms[i]+')\\b', 'gi'), "<span class='yellow'>$1</span>" );
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
});
