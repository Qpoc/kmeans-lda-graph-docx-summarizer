$(document).ready(function () { 
    var files = document.getElementById("IP_doc_files");
    var numberSentence = document.getElementById("numberSentence");
    var numberCluster = document.getElementById("numberCluster");

    $("#isRemoveRedundant").change(function (e) { 
        
        if ($("#isRemoveRedundant").val() == "on") {
            $("#isRemoveRedundant").val("off")
            $("#removeRedundant").val("False")
        }else{
            $("#isRemoveRedundant").val("on")
            $("#removeRedundant").val("True")
        }
    })

    numberCluster.addEventListener("change", function (e) {
        document.getElementById("spanNumberCluster").innerText = e.target.value;
    });

    $(document).on('click','.nav-link.active', (e) =>{
        const nav = $(e.target);
        const textAreaID = nav.attr('data-bs-target');
        const text = $(`${textAreaID}`).text();
        
        $("#numChars").text(text.length);
        $("#numWords").text(countWords(text));
        // $("#numSentence").text(countSentence(text))

    });

    $(document).on('change keyup click','textarea', (e) =>{
        const nav = $(e.target);
        const text = nav.val();
        
        $("#numChars").text(text.length);
        $("#numWords").text(countWords(text));
        // $("#numSentence").text(countSentence(text))
    });

    files.addEventListener("change", function(e) {
        document.getElementById("pills-tab").innerHTML = "";
        document.getElementById("pills-tabContent").innerHTML = `<div class="spinner-border text-primary m-auto" role="status">
        </div>`;

        document.getElementById("form_doc_files").submit();
        
    })

    $('#isDefaultClustering').change(function() {
        if ($(this).prop('checked') == false) {
            $("#cluster-container").html(`<label for="numberSentence" class="form-label text-secondary" style="font-size: 15px;">
            Select number of clusters
        </label>
        <select class="form-select form-select-sm" name="numberCluster" id="numberCluster">
            <option value="4" disabled>Select number of clusters</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4" selected>4</option>
            <option value="5">5</option>
        </select>`);

        }else {
            $("#cluster-container").html(`<label for="numberCluster" class="form-label small" hidden>Number of Clusters: <span id="spanNumberCluster">4</span></label >
            <input type="text" class="form-range" name="numberCluster" id="numberCluster" min="2" step="1" max="5" value="auto" hidden/>`);
        }
    })

    function countWords(s){
        s = s.replace(/(^\s*)|(\s*$)/gi,"");//exclude  start and end white-space
        s = s.replace(/[ ]{2,}/gi," ");//2 or more space to 1
        s = s.replace(/\n /,"\n"); // exclude newline with a start spacing
        return s.split(' ').filter(function(str){return str!="";}).length;
        //return s.split(' ').filter(String).length; - this can also be used
    }

    function countSentence(s){

        return s.split('.').filter(function(str){return str!="";}).length;
    }

    $(".nav-link.active").click();
})