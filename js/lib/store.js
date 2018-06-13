

function initDBstore(sn) {
	  var indexedDBOpenRequest;

    indexedDBOpenRequest = indexedDB.open(sn);
  // This top-level error handler will be invoked any time there's an IndexedDB-related error.
  indexedDBOpenRequest.onerror = function(error) {
      
    console.error('IndexedDB error:', error);
     // indexedDB.createObjectStore(STORE_NAME);
  };

  // This should only execute if there's a need to create a new database for the given IDB_VERSION.
  indexedDBOpenRequest.onupgradeneeded = function() {
  //  this.result.createObjectStore(STORE_NAME, {keyPath: 'url'});
      if(sn=='offana'){
          this.result.createObjectStore(sn, {keyPath: 'url'});
      }else{
       
    this.result.createObjectStore(sn);   
      }
  };

  // This will execute each time the database is opened.
  indexedDBOpenRequest.onsuccess = function() {
      if(sn=='data'){idbDatabase = this.result;}else if(sn=='notes'){notesDatabase = this.result;}else if(sn=='offana'){anaDatabase = this.result;}
   //
	  indexDbReady = new Promise((resolve, reject) => {
  	resolve('ready');
}); 
  };
}
  
// Helper method to get the object store that we care about.
function getAnaStore(storeName) {
    
  return anaDatabase.transaction('offana', 'readwrite').objectStore(storeName);
    
        
}

function getNotesStore(storeName) {
    
  return notesDatabase.transaction('notes', 'readwrite').objectStore(storeName);
    
        
}


function getObjectStore(storeName) {
    
    
  return idbDatabase.transaction('data', 'readwrite').objectStore(storeName);
    
        
}



function DBstore(storeName, data) {
  getObjectStore(storeName, 'readwrite').add({
    url: data
  });
}


	
var idbDatabase;
var IDB_VERSION = 1;
 //var indexedDB = window.indexedDB || window.webkitIndexedDB || window.mozIndexedDB || window.OIndexedDB || window.msIndexedDB,
    //    IDBTransaction = window.IDBTransaction || window.webkitIDBTransaction || window.OIDBTransaction || window.msIDBTransaction;

    	initDBstore('data');
indexDbReady = Promise.race([]);