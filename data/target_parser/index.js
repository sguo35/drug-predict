var csv = require('csv');
var fs = require('fs');

Array.prototype.remove = function (from, to) {
    var rest = this.slice((to || from) + 1 || this.length);
    this.length = from < 0 ? this.length + from : from;
    return this.push.apply(this, rest);
};

const parseTargets = function () {
    console.log("Importing csvs...");
    // import CSV files to be parsed
    csv.parse(fs.readFileSync('../all.csv'), function (err, data) {
        if (err) console.log(err);
        // look through each row and delete irrelevant data
        for (let row of data) {
            row.remove(3); // GenBank Protein ID
            row.remove(3); // GenBank Gene ID
            row.remove(3); // UniProt ID
            row.remove(3); // Uniprot Title
            row.remove(3); // PDB ID
            row.remove(3); // GeneCard ID
            row.remove(3); // GenAtlas ID
            row.remove(3); // HGNC ID
            row.remove(3); // Species

            // remove the semicolons and place drugs into their own array
            let unseparatedDrugs = row[3];
            while (unseparatedDrugs.indexOf(";") >= 0) {
                unseparatedDrugs = unseparatedDrugs.replace(";", "");
            }
            row[3] = unseparatedDrugs.split(" ");

            // remove name and gene name
            row.remove(2); // Name
            row.remove(1); // Gene name
        }

        // Place each drug target pair in its own row
        let newDTPs = [];
        for (let row of data) {
            for (let drug of row[1]) {
                newDTPs.push([row[0], drug]);
            }
        }
        console.log(newDTPs);
        csv.stringify(newDTPs, function (err, data) {
            if (err) console.log(err);
            fs.writeFileSync('../experimental_target_pairs.csv', data);
        })
    });
}

parseTargets();